# CUDA frames (g2g backend)
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from functools import cache

LIMITED_RANGE_LUMA_OFFSET = 16
FULL_RANGE_LUMA_OFFSET = 0
LIMITED_RANGE_LUMA_LEVELS = 219
LIMITED_RANGE_CHROMA_LEVELS = 224
CHROMA_OFFSET = 128
CHROMA_SUBSAMPLING = 2
MAXIMUM_LEVEL = 255
CHROMA_CHANNELS = 2
OPAQUE_ALPHA = MAXIMUM_LEVEL


def luma_offset(yuv_conversion):
    _, _, full_range = yuv_conversion
    return FULL_RANGE_LUMA_OFFSET if full_range else LIMITED_RANGE_LUMA_OFFSET


def rgb_from_ycbcr(yuv_conversion):
    red_weight, blue_weight, full_range = yuv_conversion
    green_weight = 1 - red_weight - blue_weight
    luma_scale = 1 if full_range else MAXIMUM_LEVEL / LIMITED_RANGE_LUMA_LEVELS
    chroma_scale = 1 if full_range else MAXIMUM_LEVEL / LIMITED_RANGE_CHROMA_LEVELS
    red_from_red_difference = 2 * (1 - red_weight) * chroma_scale
    blue_from_blue_difference = 2 * (1 - blue_weight) * chroma_scale
    green_from_blue_difference = (
        2 * blue_weight * (1 - blue_weight) / green_weight * chroma_scale
    )
    green_from_red_difference = (
        2 * red_weight * (1 - red_weight) / green_weight * chroma_scale
    )
    return (
        (luma_scale, 0, red_from_red_difference),
        (luma_scale, -green_from_blue_difference, -green_from_red_difference),
        (luma_scale, blue_from_blue_difference, 0),
    )


def nv12_to_rgb(luma, chroma, yuv_conversion):
    import torch

    height, width = luma.shape
    chroma_differences = (
        (chroma.float() - CHROMA_OFFSET)
        .repeat_interleave(CHROMA_SUBSAMPLING, dim=0)
        .repeat_interleave(CHROMA_SUBSAMPLING, dim=1)[:height, :width]
    )
    blue_difference, red_difference = chroma_differences.unbind(dim=-1)
    offset_luma = luma.float() - luma_offset(yuv_conversion)
    rgb = torch.stack(
        [
            luma_coefficient * offset_luma
            + blue_difference_coefficient * blue_difference
            + red_difference_coefficient * red_difference
            for (
                luma_coefficient,
                blue_difference_coefficient,
                red_difference_coefficient,
            ) in rgb_from_ycbcr(yuv_conversion)
        ],
        dim=-1,
    )
    return rgb.round_().clamp_(0, MAXIMUM_LEVEL).to(torch.uint8)


def rgb_device_frame(luma_plane, chroma_plane, yuv_conversion):
    import torch

    # torch refuses the read-only array interface the planes export
    return nv12_to_rgb(
        torch.from_dlpack(luma_plane), torch.from_dlpack(chroma_plane), yuv_conversion
    )


@cache
def ycbcr_from_rgb_matrix(device, yuv_conversion):
    import torch

    matrix = torch.tensor(rgb_from_ycbcr(yuv_conversion), dtype=torch.float64)
    return torch.linalg.inv(matrix).float().to(device)


def as_levels(values):
    import torch

    return values.round_().clamp_(0, MAXIMUM_LEVEL).to(torch.uint8)


def rgb_to_nv12(rgb, yuv_conversion):
    import torch

    height, width, _ = rgb.shape
    ycbcr = rgb.float() @ ycbcr_from_rgb_matrix(rgb.device, yuv_conversion).T
    luma = ycbcr[..., 0] + luma_offset(yuv_conversion)
    chroma_blocks = ycbcr[..., 1:].reshape(
        height // CHROMA_SUBSAMPLING,
        CHROMA_SUBSAMPLING,
        width // CHROMA_SUBSAMPLING,
        CHROMA_SUBSAMPLING,
        CHROMA_CHANNELS,
    )
    chroma = chroma_blocks.mean(dim=(1, 3)) + CHROMA_OFFSET
    # nvenc registers the luma pointer alone and reads chroma right after it
    surface = torch.empty(
        (height + height // CHROMA_SUBSAMPLING, width),
        dtype=torch.uint8,
        device=rgb.device,
    )
    surface[:height] = as_levels(luma)
    surface[height:].view(chroma.shape).copy_(as_levels(chroma))
    return surface[:height], surface[height:].view(chroma.shape)


def rgba_device_frame(luma_plane, chroma_plane, yuv_conversion):
    import torch

    rgb = rgb_device_frame(luma_plane, chroma_plane, yuv_conversion)
    alpha = torch.full_like(rgb[..., :1], OPAQUE_ALPHA)
    return torch.cat((rgb, alpha), dim=-1)
