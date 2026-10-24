# Frame format conversion
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Shared, framework-agnostic pixel-format conversion for task code.

Tasks that draw overlays work in RGB and convert to the target video pixel
format before handing the frame back to the element shell for writing.
"""


def to_format(rgb, fmt):
    """Convert an RGB numpy array to the target video pixel format `fmt`."""
    import cv2
    import numpy as np

    if fmt == "RGB":
        return rgb
    elif fmt == "BGR":
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif fmt == "RGBA":
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
    elif fmt == "BGRA":
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGRA)
    elif fmt == "ARGB":
        rgba = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
        return np.roll(rgba, 1, axis=-1)
    elif fmt == "ABGR":
        bgra = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGRA)
        return np.roll(bgra, 1, axis=-1)
    else:
        return rgb


# the buffer keeps its alpha channel, the write fills the colour channels only
def to_device_format(rgb, fmt):
    from utils.format_converter import packed_format_layout

    _, rgb_slice = packed_format_layout(fmt)
    reversed_channels = (rgb_slice[-1].step or 1) < 0
    return rgb.flip(-1) if reversed_channels else rgb
