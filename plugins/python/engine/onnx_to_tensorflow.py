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
# options for the onnx2tf 1.x line that ultralytics installs on its own exports
ONNX2TF_OPTIONS = {
    "output_signaturedefs": True,
    "not_use_onnxsim": True,
    "disable_strict_mode": True,
    "verbosity": "error",
}
FLOAT32_TFLITE_SUFFIX = "_float32.tflite"
SAVED_MODEL_OUTPUT_PREFIX = "output_"


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


# onnx2tf names the signature outputs output_0, output_1... in onnx order
def saved_model_output_names(onnx_path):
    return [
        f"{SAVED_MODEL_OUTPUT_PREFIX}{index}"
        for index in range(len(onnx_output_names(onnx_path)))
    ]


# one conversion writes the SavedModel and the float32 tflite side by side
def write_conversion(onnx_path, output_path):
    import onnx2tf

    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(output_path),
        keep_shape_absolutely_input_names=onnx_input_names(onnx_graph(onnx_path)),
        **ONNX2TF_OPTIONS,
    )


def cached_conversion(onnx_path):
    conversion_path = converted_model_path(
        CONVERSION_CACHE_NAME,
        onnx_path,
        CONVERSION_DIRECTORY_SUFFIX,
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
    return cached_conversion(onnx_path)


def float32_tflite_from_onnx(onnx_path):
    return (
        cached_conversion(onnx_path) / f"{Path(onnx_path).stem}{FLOAT32_TFLITE_SUFFIX}"
    )
