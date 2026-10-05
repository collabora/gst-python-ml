# ONNX export cache
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Library General Public License for more details.
#
# You should have received a copy of the GNU Library General Public
# License along with this library; if not, write to the
# Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
# Boston, MA 02110-1301, USA.

from pathlib import Path

ONNX_EXPORT_CACHE = Path.home() / ".cache" / "gst-python-ml" / "onnx"
PARTIAL_EXPORT_SUFFIX = ".partial"
GRAPH_INPUT_NAME = "image"
GRAPH_OUTPUT_NAME = "output"
ANY_FRAME_SHAPE = (480, 640, 3)
UINT8_MAX = 255
# large enough to fold a vision transformer's position embeddings
CONSTANT_FOLDING_INPUT_SIZE_LIMIT = 1 << 20
RESIZE_POLICY_ATTRIBUTE = "keep_aspect_ratio_policy"
DEFAULT_RESIZE_POLICY = "stretch"


# a builtin engine feeds the frame scaled to 0..1
def pixel_normalizer(image_mean, image_std):
    import torch

    channel_shape = (1, -1, 1, 1)
    mean = torch.tensor(image_mean).view(channel_shape)
    std = torch.tensor(image_std).view(channel_shape)
    return lambda image: (image - mean) / std


# resized and cropped the way the model was trained
def model_input_frames(image_processor, frames):
    import numpy as np

    pixel_values = image_processor(frames, do_normalize=False, return_tensors="np")[
        "pixel_values"
    ][0]
    # a builtin engine divides by 255 again
    return np.ascontiguousarray(np.moveaxis(pixel_values, -3, -1)) * UINT8_MAX


def model_input_shape(image_processor, frame_count=None):
    import numpy as np

    any_frame = np.zeros(ANY_FRAME_SHAPE, dtype=np.uint8)
    frames = any_frame if frame_count is None else [any_frame] * frame_count
    return model_input_frames(image_processor, frames).shape


# migraphx rejects the attribute even at its default value
def remove_default_resize_policy(model):
    for node in model.graph.all_nodes():
        policy = node.attributes.get(RESIZE_POLICY_ATTRIBUTE)
        if policy is not None and policy.value == DEFAULT_RESIZE_POLICY:
            del node.attributes[RESIZE_POLICY_ATTRIBUTE]


def cached_onnx_export(file_stem, build_graph, **export_options):
    import onnxscript.optimizer
    import torch

    path = ONNX_EXPORT_CACHE / f"{file_stem}.onnx"
    if path.exists():
        return str(path)
    ONNX_EXPORT_CACHE.mkdir(parents=True, exist_ok=True)
    graph, example_input = build_graph()
    program = torch.onnx.export(
        graph.eval(),
        (example_input,),
        input_names=[GRAPH_INPUT_NAME],
        output_names=[GRAPH_OUTPUT_NAME],
        **export_options,
    )
    onnxscript.optimizer.optimize(
        program.model, input_size_limit=CONSTANT_FOLDING_INPUT_SIZE_LIMIT
    )
    remove_default_resize_policy(program.model)
    # an interrupted export would leave a file the next load takes for the model
    partial_path = path.with_suffix(PARTIAL_EXPORT_SUFFIX)
    # a weights file beside the model would keep the partial file's name
    program.save(str(partial_path), external_data=False)
    partial_path.rename(path)
    return str(path)
