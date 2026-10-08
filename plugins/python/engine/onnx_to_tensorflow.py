# ONNX to TensorFlow conversion
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

import os
import tempfile
from pathlib import Path

from .ml_engine import converted_model_path

ONNX_SUFFIX = ".onnx"
CONVERSION_CACHE_NAME = "tensorflow"
CONVERSION_DIRECTORY_SUFFIX = ""
CONVERSION_OUTPUT_NAME = "converted"
# the flatbuffer_direct backend writes no SavedModel for gelu or depthwise convolutions
SAVED_MODEL_BACKEND = "tf_converter"
TFLITE_BACKEND = "flatbuffer_direct"
ONNX2TF_OPTIONS = {
    "not_use_onnxsim": True,
    "disable_strict_mode": True,
    "verbosity": "error",
}
FLOAT32_TFLITE_SUFFIX = "_float32.tflite"


def onnx_graph(onnx_path):
    import onnx

    return onnx.load(onnx_path, load_external_data=False).graph


def onnx_input_names(graph):
    initializer_names = {initializer.name for initializer in graph.initializer}
    return [
        graph_input.name
        for graph_input in graph.input
        if graph_input.name not in initializer_names
    ]


def onnx_output_names(onnx_path):
    return [graph_output.name for graph_output in onnx_graph(onnx_path).output]


def onnx2tf_convert(onnx_path, output_path, **options):
    import onnx2tf

    return onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(output_path),
        # the tf_converter backend ignores keep_ncw_or_nchw_or_ncdhw_input_names
        keep_shape_absolutely_input_names=onnx_input_names(onnx_graph(onnx_path)),
        **ONNX2TF_OPTIONS,
        **options,
    )


# tf_converter would also spend minutes writing two tflite files nobody loads
def write_saved_model(onnx_path, output_path):
    import tensorflow as tf

    keras_model = onnx2tf_convert(
        onnx_path,
        output_path,
        tflite_backend=SAVED_MODEL_BACKEND,
        disable_model_save=True,
    )
    tf.saved_model.save(keras_model, str(output_path))


def write_tflite(onnx_path, output_path):
    onnx2tf_convert(onnx_path, output_path, tflite_backend=TFLITE_BACKEND)


def cached_conversion(onnx_path, write_conversion):
    conversion_path = converted_model_path(
        CONVERSION_CACHE_NAME,
        onnx_path,
        CONVERSION_DIRECTORY_SUFFIX,
        write_conversion.__name__,
        *sorted(ONNX2TF_OPTIONS.items()),
    )
    if conversion_path.is_dir():
        return conversion_path

    conversion_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=conversion_path.parent) as work_directory:
        output_path = Path(work_directory) / CONVERSION_OUTPUT_NAME
        write_conversion(onnx_path, output_path)
        os.replace(output_path, conversion_path)
    return conversion_path


def saved_model_from_onnx(onnx_path):
    return cached_conversion(onnx_path, write_saved_model)


def float32_tflite_from_onnx(onnx_path):
    conversion_path = cached_conversion(onnx_path, write_tflite)
    return conversion_path / f"{Path(onnx_path).stem}{FLOAT32_TFLITE_SUFFIX}"
