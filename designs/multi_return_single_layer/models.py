"""Models for selecting at most one point from K returns per range-image pixel.

Input conventions:
    points:     [B, K, F, H, W]
    ranges:     [B, K, H, W], sorted from near to far
    valid_mask: [B, K, H, W], True where a candidate exists

Selection conventions:
    0 means that no point is selected.
    1..K select candidate layers 0..K-1 respectively.

The context backbone is CENet-inspired rather than a verbatim copy of the
official CENet implementation. It uses large-kernel context blocks and a
lightweight encoder-decoder while keeping the code self-contained.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        activate: bool = True,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        modules = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        ]
        if activate:
            modules.append(nn.ReLU(inplace=True))
        super().__init__(*modules)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(channels, channels, 3),
            ConvGNAct(channels, channels, 3, activate=False),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.block(x))


class LargeKernelContextBlock(nn.Module):
    """Efficient large-kernel context block for range-view features."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.local = ConvGNAct(
            channels, channels, kernel_size=7, groups=channels, activate=False
        )
        self.horizontal = ConvGNAct(
            channels,
            channels,
            kernel_size=1,
            groups=channels,
            activate=False,
        )
        self.horizontal_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, 11),
            padding=(0, 5),
            groups=channels,
            bias=False,
        )
        self.vertical_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(11, 1),
            padding=(5, 0),
            groups=channels,
            bias=False,
        )
        self.project = ConvGNAct(channels, channels, 1, activate=False)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        local = self.local(x)
        axial = self.horizontal(x)
        axial = self.horizontal_conv(axial)
        axial = self.vertical_conv(axial)
        return self.activation(x + self.project(local + axial))


class CENetStyleContextBackbone(nn.Module):
    """CENet-style encoder-decoder that returns full-resolution features."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        out_channels: int = 64,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 6

        self.stem = ConvGNAct(in_channels, c1, 3)
        self.enc1 = ResidualBlock(c1)
        self.down1 = ConvGNAct(c1, c2, 3, stride=2)
        self.enc2 = ResidualBlock(c2)
        self.down2 = ConvGNAct(c2, c3, 3, stride=2)
        self.enc3 = nn.Sequential(ResidualBlock(c3), LargeKernelContextBlock(c3))
        self.down3 = ConvGNAct(c3, c4, 3, stride=2)
        self.bottleneck = nn.Sequential(
            ResidualBlock(c4),
            LargeKernelContextBlock(c4),
        )

        self.dec3 = nn.Sequential(ConvGNAct(c4 + c3, c3, 3), ResidualBlock(c3))
        self.dec2 = nn.Sequential(ConvGNAct(c3 + c2, c2, 3), ResidualBlock(c2))
        self.dec1 = nn.Sequential(ConvGNAct(c2 + c1, c1, 3), ResidualBlock(c1))
        self.output = ConvGNAct(c1, out_channels, 1)

    @staticmethod
    def _resize(x: Tensor, reference: Tensor) -> Tensor:
        return F.interpolate(
            x,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        e1 = self.enc1(self.stem(x))
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        x = self.bottleneck(self.down3(e3))

        x = self.dec3(torch.cat([self._resize(x, e3), e3], dim=1))
        x = self.dec2(torch.cat([self._resize(x, e2), e2], dim=1))
        x = self.dec1(torch.cat([self._resize(x, e1), e1], dim=1))
        return self.output(x)


class MultiReturnEncoder(nn.Module):
    """Shared point encoding followed by joint layer-context extraction."""

    def __init__(
        self,
        point_channels: int,
        num_layers: int = 3,
        point_feature_channels: int = 32,
        context_channels: int = 64,
        candidate_channels: int = 64,
        backbone_base_channels: int = 32,
        range_scale: float = 100.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")

        self.point_channels = point_channels
        self.num_layers = num_layers
        self.point_feature_channels = point_feature_channels
        self.context_channels = context_channels
        self.candidate_channels = candidate_channels
        self.range_scale = range_scale

        self.point_stem = nn.Sequential(
            ConvGNAct(point_channels, point_feature_channels, 1),
            ConvGNAct(point_feature_channels, point_feature_channels, 1),
        )

        # masks(K) + count(1) + normalized ranges(K) + adjacent gaps(K-1)
        geometry_channels = 3 * num_layers
        context_in_channels = num_layers * point_feature_channels + geometry_channels
        self.context_backbone = CENetStyleContextBackbone(
            context_in_channels,
            base_channels=backbone_base_channels,
            out_channels=context_channels,
        )

        # point feature + common context + range + relative range + mask + rank
        candidate_in_channels = point_feature_channels + context_channels + 4
        self.candidate_fuser = nn.Sequential(
            ConvGNAct(candidate_in_channels, candidate_channels, 1),
            ConvGNAct(candidate_channels, candidate_channels, 3),
        )

    def _validate_inputs(
        self, points: Tensor, ranges: Tensor, valid_mask: Tensor
    ) -> Tuple[int, int, int, int, int]:
        if points.ndim != 5:
            raise ValueError("points must have shape [B, K, F, H, W]")
        batch, layers, channels, height, width = points.shape
        if layers != self.num_layers:
            raise ValueError(f"expected K={self.num_layers}, got K={layers}")
        if channels != self.point_channels:
            raise ValueError(
                f"expected F={self.point_channels}, got F={channels}"
            )
        if ranges.shape != (batch, layers, height, width):
            raise ValueError("ranges must have shape [B, K, H, W]")
        if valid_mask.shape != ranges.shape:
            raise ValueError("valid_mask must have shape [B, K, H, W]")
        return batch, layers, channels, height, width

    def _geometry(
        self, ranges: Tensor, valid_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        mask = valid_mask.to(dtype=ranges.dtype)
        normalized_ranges = torch.where(
            valid_mask,
            ranges.clamp_min(0.0) / self.range_scale,
            torch.zeros_like(ranges),
        )
        count = mask.sum(dim=1, keepdim=True) / float(self.num_layers)

        if self.num_layers > 1:
            adjacent_valid = valid_mask[:, 1:] & valid_mask[:, :-1]
            adjacent_gaps = normalized_ranges[:, 1:] - normalized_ranges[:, :-1]
            adjacent_gaps = torch.where(
                adjacent_valid,
                adjacent_gaps,
                torch.zeros_like(adjacent_gaps),
            )
        else:
            adjacent_gaps = normalized_ranges.new_zeros(
                normalized_ranges.shape[0], 0, *normalized_ranges.shape[-2:]
            )

        geometry = torch.cat(
            [mask, count, normalized_ranges, adjacent_gaps], dim=1
        )
        return geometry, normalized_ranges

    def forward(
        self, points: Tensor, ranges: Tensor, valid_mask: Tensor
    ) -> Dict[str, Tensor]:
        batch, layers, channels, height, width = self._validate_inputs(
            points, ranges, valid_mask
        )
        valid_mask = valid_mask.bool()
        masked_points = points * valid_mask.unsqueeze(2).to(points.dtype)

        point_features = self.point_stem(
            masked_points.reshape(batch * layers, channels, height, width)
        )
        point_features = point_features.reshape(
            batch,
            layers,
            self.point_feature_channels,
            height,
            width,
        )

        geometry, normalized_ranges = self._geometry(ranges, valid_mask)
        context_input = torch.cat(
            [point_features.flatten(1, 2), geometry], dim=1
        )
        context = self.context_backbone(context_input)

        context_per_candidate = context.unsqueeze(1).expand(
            -1, layers, -1, -1, -1
        )
        nearest_range = normalized_ranges[:, :1]
        relative_range = normalized_ranges - nearest_range
        rank = torch.linspace(
            0.0,
            1.0,
            layers,
            device=points.device,
            dtype=points.dtype,
        ).view(1, layers, 1, 1, 1)
        rank = rank.expand(batch, -1, -1, height, width)

        candidate_input = torch.cat(
            [
                point_features,
                context_per_candidate,
                normalized_ranges.unsqueeze(2),
                relative_range.unsqueeze(2),
                valid_mask.unsqueeze(2).to(points.dtype),
                rank,
            ],
            dim=2,
        )
        candidate_features = self.candidate_fuser(
            candidate_input.reshape(
                batch * layers, candidate_input.shape[2], height, width
            )
        )
        candidate_features = candidate_features.reshape(
            batch, layers, self.candidate_channels, height, width
        )
        candidate_features = candidate_features * valid_mask.unsqueeze(2).to(
            candidate_features.dtype
        )

        return {
            "candidate_features": candidate_features,
            "context": context,
            "normalized_ranges": normalized_ranges,
            "valid_mask": valid_mask,
        }


def gather_selected_points(
    points: Tensor, selection: Tensor
) -> Tuple[Tensor, Tensor]:
    """Gather a single-layer point map from 0..K selection indices."""

    if points.ndim != 5 or selection.ndim != 3:
        raise ValueError("points must be [B,K,F,H,W] and selection [B,H,W]")
    batch, layers, channels, height, width = points.shape
    if selection.shape != (batch, height, width):
        raise ValueError("selection spatial shape does not match points")

    output_mask = selection > 0
    candidate_index = (selection - 1).clamp(min=0, max=layers - 1)
    gather_index = candidate_index[:, None, None].expand(
        -1, 1, channels, -1, -1
    )
    selected_points = points.gather(dim=1, index=gather_index).squeeze(1)
    selected_points = selected_points * output_mask.unsqueeze(1).to(points.dtype)
    return selected_points, output_mask


class BinaryNearestModel(nn.Module):
    """Method 1: binary candidate classification plus nearest-target rule."""

    def __init__(self, point_channels: int, num_layers: int = 3, **encoder_kwargs) -> None:
        super().__init__()
        self.encoder = MultiReturnEncoder(
            point_channels=point_channels,
            num_layers=num_layers,
            **encoder_kwargs,
        )
        self.binary_head = nn.Conv2d(
            self.encoder.candidate_channels, 1, kernel_size=1
        )

    def forward(
        self, points: Tensor, ranges: Tensor, valid_mask: Tensor
    ) -> Dict[str, Tensor]:
        encoded = self.encoder(points, ranges, valid_mask)
        features = encoded["candidate_features"]
        batch, layers, channels, height, width = features.shape
        logits = self.binary_head(
            features.reshape(batch * layers, channels, height, width)
        ).reshape(batch, layers, height, width)
        return {"candidate_logits": logits}

    @torch.no_grad()
    def predict(
        self,
        points: Tensor,
        ranges: Tensor,
        valid_mask: Tensor,
        threshold: float = 0.5,
    ) -> Dict[str, Tensor]:
        logits = self(points, ranges, valid_mask)["candidate_logits"]
        probabilities = torch.sigmoid(logits)
        keep = (probabilities >= threshold) & valid_mask.bool()
        selectable_ranges = ranges.masked_fill(~keep, torch.inf)
        candidate_index = selectable_ranges.argmin(dim=1)
        has_target = keep.any(dim=1)
        selection = torch.where(
            has_target, candidate_index + 1, torch.zeros_like(candidate_index)
        )
        selected_points, output_mask = gather_selected_points(points, selection)
        return {
            "selected_points": selected_points,
            "output_mask": output_mask,
            "selection": selection,
            "candidate_probabilities": probabilities,
        }


class JointSelectionModel(nn.Module):
    """Method 2: learned K+1 selection over none and all candidates."""

    def __init__(self, point_channels: int, num_layers: int = 3, **encoder_kwargs) -> None:
        super().__init__()
        self.encoder = MultiReturnEncoder(
            point_channels=point_channels,
            num_layers=num_layers,
            **encoder_kwargs,
        )
        self.candidate_selection_head = nn.Conv2d(
            self.encoder.candidate_channels, 1, kernel_size=1
        )
        self.none_head = nn.Sequential(
            ConvGNAct(self.encoder.context_channels, 32, 3),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def _selection_logits(self, encoded: Dict[str, Tensor]) -> Tensor:
        features = encoded["candidate_features"]
        valid_mask = encoded["valid_mask"]
        batch, layers, channels, height, width = features.shape
        candidate_logits = self.candidate_selection_head(
            features.reshape(batch * layers, channels, height, width)
        ).reshape(batch, layers, height, width)
        candidate_logits = candidate_logits.masked_fill(~valid_mask, -1e4)
        none_logit = self.none_head(encoded["context"])
        return torch.cat([none_logit, candidate_logits], dim=1)

    def forward(
        self, points: Tensor, ranges: Tensor, valid_mask: Tensor
    ) -> Dict[str, Tensor]:
        encoded = self.encoder(points, ranges, valid_mask)
        return {"selection_logits": self._selection_logits(encoded)}

    @torch.no_grad()
    def predict(
        self, points: Tensor, ranges: Tensor, valid_mask: Tensor
    ) -> Dict[str, Tensor]:
        logits = self(points, ranges, valid_mask)["selection_logits"]
        probabilities = torch.softmax(logits, dim=1)
        selection = probabilities.argmax(dim=1)
        selected_points, output_mask = gather_selected_points(points, selection)
        return {
            "selected_points": selected_points,
            "output_mask": output_mask,
            "selection": selection,
            "selection_probabilities": probabilities,
        }


class JointSelectionAuxModel(JointSelectionModel):
    """Method 3: K+1 selection with auxiliary candidate binary supervision."""

    def __init__(self, point_channels: int, num_layers: int = 3, **encoder_kwargs) -> None:
        super().__init__(point_channels, num_layers, **encoder_kwargs)
        self.candidate_binary_head = nn.Conv2d(
            self.encoder.candidate_channels, 1, kernel_size=1
        )

    def forward(
        self, points: Tensor, ranges: Tensor, valid_mask: Tensor
    ) -> Dict[str, Tensor]:
        encoded = self.encoder(points, ranges, valid_mask)
        features = encoded["candidate_features"]
        batch, layers, channels, height, width = features.shape
        candidate_logits = self.candidate_binary_head(
            features.reshape(batch * layers, channels, height, width)
        ).reshape(batch, layers, height, width)
        return {
            "selection_logits": self._selection_logits(encoded),
            "candidate_logits": candidate_logits,
        }

    @torch.no_grad()
    def predict(
        self, points: Tensor, ranges: Tensor, valid_mask: Tensor
    ) -> Dict[str, Tensor]:
        outputs = self(points, ranges, valid_mask)
        probabilities = torch.softmax(outputs["selection_logits"], dim=1)
        selection = probabilities.argmax(dim=1)
        selected_points, output_mask = gather_selected_points(points, selection)
        return {
            "selected_points": selected_points,
            "output_mask": output_mask,
            "selection": selection,
            "selection_probabilities": probabilities,
            "candidate_probabilities": torch.sigmoid(
                outputs["candidate_logits"]
            ),
        }


def masked_binary_focal_loss(
    logits: Tensor,
    targets: Tensor,
    valid_mask: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    """Binary focal loss over valid candidate points only."""

    targets = targets.to(logits.dtype)
    valid = valid_mask.to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = torch.sigmoid(logits)
    pt = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * (1.0 - pt).pow(gamma) * bce * valid
    return loss.sum() / valid.sum().clamp_min(1.0)


def selection_cross_entropy(
    selection_logits: Tensor,
    selection_labels: Tensor,
    class_weights: Optional[Tensor] = None,
) -> Tensor:
    """Cross entropy for labels 0=none and 1..K=candidate index + 1."""

    return F.cross_entropy(
        selection_logits,
        selection_labels.long(),
        weight=class_weights,
    )


def joint_selection_aux_loss(
    outputs: Dict[str, Tensor],
    selection_labels: Tensor,
    candidate_labels: Tensor,
    valid_mask: Tensor,
    selection_class_weights: Optional[Tensor] = None,
    auxiliary_weight: float = 0.2,
) -> Dict[str, Tensor]:
    selection_loss = selection_cross_entropy(
        outputs["selection_logits"],
        selection_labels,
        selection_class_weights,
    )
    candidate_loss = masked_binary_focal_loss(
        outputs["candidate_logits"], candidate_labels, valid_mask
    )
    total = selection_loss + auxiliary_weight * candidate_loss
    return {
        "loss": total,
        "selection_loss": selection_loss,
        "candidate_loss": candidate_loss,
    }


def build_model(
    method: str,
    point_channels: int,
    num_layers: int = 3,
    **encoder_kwargs,
) -> nn.Module:
    methods = {
        "binary_nearest": BinaryNearestModel,
        "joint_selection": JointSelectionModel,
        "joint_selection_aux": JointSelectionAuxModel,
    }
    if method not in methods:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(methods)}")
    return methods[method](
        point_channels=point_channels,
        num_layers=num_layers,
        **encoder_kwargs,
    )
