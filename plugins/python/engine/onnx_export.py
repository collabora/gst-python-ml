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

from .engine_factory import EngineFactory
from .ml_engine import MODEL_CACHE

ONNX_EXPORT_CACHE = MODEL_CACHE / "onnx"
EXECUTORCH_EXPORT_CACHE = MODEL_CACHE / EngineFactory.EXECUTORCH_ENGINE
EXECUTORCH_SUFFIX = ".pte"
PARTIAL_EXPORT_SUFFIX = ".partial"
GRAPH_INPUT_NAME = "image"
GRAPH_OUTPUT_NAME = "output"
ANY_FRAME_SHAPE = (480, 640, 3)
UINT8_MAX = 255
# large enough to fold a vision transformer's position embeddings
CONSTANT_FOLDING_INPUT_SIZE_LIMIT = 1 << 20
RESIZE_POLICY_ATTRIBUTE = "keep_aspect_ratio_policy"
DEFAULT_RESIZE_POLICY = "stretch"
BILINEAR_RESIZE_MODE = b"linear"
ALIGN_CORNERS_TRANSFORM = b"align_corners"
RESIZE_SIZES_INPUT = 3
IMAGE_RANK = 4


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


def interpolation_matrix(source, destination):
    import numpy as np

    matrix = np.zeros((destination, source), np.float32)
    for index in range(destination):
        position = index * (source - 1) / (destination - 1)
        low = int(np.floor(position))
        weight = position - low
        matrix[index, low] += 1 - weight
        if low + 1 < source:
            matrix[index, low + 1] += weight
    return matrix


def static_float_shapes(graph):
    import onnx

    shapes = {}
    for info in [*graph.input, *graph.value_info]:
        tensor_type = info.type.tensor_type
        dims = tensor_type.shape.dim
        is_static = all(dim.HasField("dim_value") for dim in dims)
        if tensor_type.elem_type == onnx.TensorProto.FLOAT and is_static:
            shapes[info.name] = [dim.dim_value for dim in dims]
    return shapes


def bilinear_resize_as_matmuls(node, float_shapes, initializers):
    import onnx
    from onnx import numpy_helper

    if node.op_type != "Resize" or len(node.input) <= RESIZE_SIZES_INPUT:
        return None
    attributes = {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }
    is_bilinear_align_corners = (
        attributes.get("mode") == BILINEAR_RESIZE_MODE
        and attributes.get("coordinate_transformation_mode") == ALIGN_CORNERS_TRANSFORM
        and not attributes.get("antialias")
        and "axes" not in attributes
    )
    sizes = initializers.get(node.input[RESIZE_SIZES_INPUT])
    source = float_shapes.get(node.input[0])
    if not is_bilinear_align_corners or sizes is None or source is None:
        return None
    destination = numpy_helper.to_array(sizes).tolist()
    if len(source) != IMAGE_RANK or destination[:2] != source[:2]:
        return None
    if 1 in destination[2:]:
        return None
    output = node.output[0]
    rows_name = f"{output}_rows_interpolation"
    columns_name = f"{output}_columns_interpolation"
    rows_resized = f"{output}_rows_resized"
    matrices = [
        numpy_helper.from_array(
            interpolation_matrix(source[2], destination[2]), rows_name
        ),
        numpy_helper.from_array(
            interpolation_matrix(source[3], destination[3]).T.copy(), columns_name
        ),
    ]
    matmuls = [
        onnx.helper.make_node("MatMul", [rows_name, node.input[0]], [rows_resized]),
        onnx.helper.make_node("MatMul", [rows_resized, columns_name], [output]),
    ]
    return matmuls, matrices


# the migraphx gpu target returns garbage for this resize inside a large graph
def resize_align_corners_as_matmuls(path):
    import onnx

    model = onnx.load(path)
    graph = model.graph
    float_shapes = static_float_shapes(onnx.shape_inference.infer_shapes(model).graph)
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    nodes = []
    replaced_sizes = set()
    for node in graph.node:
        replacement = bilinear_resize_as_matmuls(node, float_shapes, initializers)
        if replacement is None:
            nodes.append(node)
            continue
        matmuls, matrices = replacement
        nodes.extend(matmuls)
        graph.initializer.extend(matrices)
        replaced_sizes.add(node.input[RESIZE_SIZES_INPUT])
    if not replaced_sizes:
        return
    used_inputs = {name for node in nodes for name in node.input}
    kept_initializers = [
        initializer
        for initializer in graph.initializer
        if initializer.name not in replaced_sizes or initializer.name in used_inputs
    ]
    graph.ClearField("node")
    graph.node.extend(nodes)
    graph.ClearField("initializer")
    graph.initializer.extend(kept_initializers)
    onnx.save(model, path)


def patch_conv_as_matmul(conv):
    import torch

    class PatchConvAsMatmul(torch.nn.Module):
        def __init__(self):
            super().__init__()
            out_channels = conv.weight.shape[0]
            self.patch_height, self.patch_width = conv.kernel_size
            self.weight = torch.nn.Parameter(
                conv.weight.detach().reshape(out_channels, -1).t().contiguous()
            )
            self.bias = None if conv.bias is None else conv.bias

        def forward(self, image):
            batch, channels, height, width = image.shape
            rows, columns = height // self.patch_height, width // self.patch_width
            patches = (
                image.reshape(
                    batch, channels * rows, self.patch_height, columns, self.patch_width
                )
                .permute(0, 1, 3, 2, 4)
                .reshape(
                    batch,
                    channels,
                    rows * columns,
                    self.patch_height * self.patch_width,
                )
                .permute(0, 2, 1, 3)
                .reshape(batch, rows * columns, -1)
            )
            embedded = patches @ self.weight
            if self.bias is not None:
                embedded = embedded + self.bias
            return embedded.reshape(batch, rows, columns, -1).permute(0, 3, 1, 2)

    return PatchConvAsMatmul()


# miopen divides by zero on a conv whose stride equals its kernel
def replace_patch_convs(module):
    import torch

    for name, child in module.named_children():
        is_patch_conv = (
            isinstance(child, torch.nn.Conv2d)
            and child.kernel_size == child.stride
            and child.kernel_size != (1, 1)
            and child.padding in ((0, 0), "valid")
            and child.groups == 1
        )
        if is_patch_conv:
            setattr(module, name, patch_conv_as_matmul(child))
        else:
            replace_patch_convs(child)


def cached_onnx_export(file_stem, build_graph, **export_options):
    import onnxscript.optimizer
    import torch

    path = ONNX_EXPORT_CACHE / f"{file_stem}.onnx"
    if path.exists():
        return str(path)
    ONNX_EXPORT_CACHE.mkdir(parents=True, exist_ok=True)
    graph, example_input = build_graph()
    replace_patch_convs(graph)
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
    resize_align_corners_as_matmuls(partial_path)
    partial_path.rename(path)
    return str(path)


def executorch_export_path(file_stem):
    return EXECUTORCH_EXPORT_CACHE / f"{file_stem}{EXECUTORCH_SUFFIX}"


def cached_executorch_export(file_stem, build_graph, **export_options):
    import torch
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower

    path = executorch_export_path(file_stem)
    if path.exists():
        return str(path)
    EXECUTORCH_EXPORT_CACHE.mkdir(parents=True, exist_ok=True)
    graph, example_input = build_graph()
    program = to_edge_transform_and_lower(
        torch.export.export(graph.eval(), (example_input,), **export_options),
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()
    partial_path = path.with_suffix(PARTIAL_EXPORT_SUFFIX)
    partial_path.write_bytes(program.buffer)
    partial_path.rename(path)
    return str(path)


# executorch has no onnx importer
def exported_model_path(engine_name, file_stem, build_graph, **export_options):
    if engine_name == EngineFactory.EXECUTORCH_ENGINE:
        return cached_executorch_export(file_stem, build_graph, **export_options)
    return cached_onnx_export(file_stem, build_graph, **export_options)
