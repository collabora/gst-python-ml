# CUDA frames (g2g backend)
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.

from functools import cache

# BT.601 limited range, the cuda hook receives no colorimetry
LUMA_OFFSET = 16
CHROMA_OFFSET = 128
LUMA_SCALE = 255 / 219
RED_FROM_RED_DIFFERENCE = 1.596
GREEN_FROM_BLUE_DIFFERENCE = 0.392
GREEN_FROM_RED_DIFFERENCE = 0.813
BLUE_FROM_BLUE_DIFFERENCE = 2.017
CHROMA_SUBSAMPLING = 2
MAXIMUM_LEVEL = 255
CHROMA_CHANNELS = 2
OPAQUE_ALPHA = MAXIMUM_LEVEL


def nv12_to_rgb(luma, chroma):
    import torch

    height, width = luma.shape
    chroma_differences = (
        (chroma.float() - CHROMA_OFFSET)
        .repeat_interleave(CHROMA_SUBSAMPLING, dim=0)
        .repeat_interleave(CHROMA_SUBSAMPLING, dim=1)[:height, :width]
    )
    blue_difference, red_difference = chroma_differences.unbind(dim=-1)
    scaled_luma = (luma.float() - LUMA_OFFSET) * LUMA_SCALE
    red = scaled_luma + RED_FROM_RED_DIFFERENCE * red_difference
    green = (
        scaled_luma
        - GREEN_FROM_BLUE_DIFFERENCE * blue_difference
        - GREEN_FROM_RED_DIFFERENCE * red_difference
    )
    blue = scaled_luma + BLUE_FROM_BLUE_DIFFERENCE * blue_difference
    rgb = torch.stack((red, green, blue), dim=-1)
    return rgb.round_().clamp_(0, MAXIMUM_LEVEL).to(torch.uint8)


def rgb_device_frame(luma_plane, chroma_plane):
    import torch

    # torch refuses the read-only array interface the planes export
    return nv12_to_rgb(torch.from_dlpack(luma_plane), torch.from_dlpack(chroma_plane))


@cache
def ycbcr_from_rgb_matrix(device):
    import torch

    rgb_from_ycbcr = torch.tensor(
        [
            [LUMA_SCALE, 0, RED_FROM_RED_DIFFERENCE],
            [LUMA_SCALE, -GREEN_FROM_BLUE_DIFFERENCE, -GREEN_FROM_RED_DIFFERENCE],
            [LUMA_SCALE, BLUE_FROM_BLUE_DIFFERENCE, 0],
        ],
        dtype=torch.float64,
    )
    return torch.linalg.inv(rgb_from_ycbcr).float().to(device)


def as_levels(values):
    import torch

    return values.round_().clamp_(0, MAXIMUM_LEVEL).to(torch.uint8)


def rgb_to_nv12(rgb):
    height, width, _ = rgb.shape
    ycbcr = rgb.float() @ ycbcr_from_rgb_matrix(rgb.device).T
    luma = ycbcr[..., 0] + LUMA_OFFSET
    chroma_blocks = ycbcr[..., 1:].reshape(
        height // CHROMA_SUBSAMPLING,
        CHROMA_SUBSAMPLING,
        width // CHROMA_SUBSAMPLING,
        CHROMA_SUBSAMPLING,
        CHROMA_CHANNELS,
    )
    chroma = chroma_blocks.mean(dim=(1, 3)) + CHROMA_OFFSET
    return as_levels(luma), as_levels(chroma)


def rgba_device_frame(luma_plane, chroma_plane):
    import torch

    rgb = rgb_device_frame(luma_plane, chroma_plane)
    alpha = torch.full_like(rgb[..., :1], OPAQUE_ALPHA)
    return torch.cat((rgb, alpha), dim=-1)
