# Engine support matrix
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

# the engines that run an exported graph, in their own format or as is
EXPORTED_MODEL_ENGINES = (
    "onnx",
    "openvino",
    "tvm",
    "tensorflow",
    "tflite",
    "ncnn",
    "executorch",
    "iree",
    "tinygrad",
    "migraphx",
)
ONNX2TF_ENGINES = ("tensorflow", "tflite")
# the package an engine's builtin path imports
ENGINE_PACKAGES = {
    "pytorch": "torch",
    "onnx": "onnxruntime",
    "openvino": "openvino",
    "tvm": "tvm",
    "tensorflow": "tensorflow",
    "tflite": "ai_edge_litert",
    "ncnn": "ncnn",
    "executorch": "executorch",
    "iree": "iree.runtime",
    "tinygrad": "tinygrad",
    "migraphx": "migraphx",
    # jax runs a keras-hub preset instead of the export
    "jax": "keras_hub",
    "llamacpp": "llama_cpp",
    "mlx": "mlx_lm",
}
SIGLIP_MODEL = "google/siglip-base-patch16-224"
DINOV2_MODEL = "facebook/dinov2-small"
# the engines each task runs on besides pytorch, as verified by the parity tests
TASK_ENGINES = {
    "yolo": EXPORTED_MODEL_ENGINES,
    "pose": EXPORTED_MODEL_ENGINES,
    "depth": EXPORTED_MODEL_ENGINES + ("jax",),
    "clip": EXPORTED_MODEL_ENGINES + ("jax",),
    "anomaly": EXPORTED_MODEL_ENGINES,
    "embedding": EXPORTED_MODEL_ENGINES,
    "action": EXPORTED_MODEL_ENGINES,
    "superres": EXPORTED_MODEL_ENGINES,
    "optical_flow": EXPORTED_MODEL_ENGINES,
    "sam": EXPORTED_MODEL_ENGINES,
    "zero_shot": EXPORTED_MODEL_ENGINES,
    "llm": ("onnx", "openvino", "llamacpp", "mlx"),
    "whisper": ("onnx", "openvino"),
}
NCNN_ONE_FRAME = "ncnn runs one frame at a time"
NCNN_BATCH_BROADCAST = "ncnn rejects the broadcast across the batch axis"
OWLV2_MEMORY = "converting owlv2 runs past 12 GB"
# why an engine refuses a task, keyed by task and engine, with the model when only one model is refused
REFUSALS = {
    ("depth", "ncnn"): NCNN_BATCH_BROADCAST,
    ("clip", "tensorflow", SIGLIP_MODEL): "onnx2tf splits a constant vector one off",
    ("clip", "tflite", SIGLIP_MODEL): "onnx2tf splits a constant vector one off",
    ("clip", "ncnn", SIGLIP_MODEL): "ncnn rejects the reshape that indexes the batch",
    ("embedding", "ncnn", DINOV2_MODEL): NCNN_BATCH_BROADCAST,
    ("action", "ncnn"): NCNN_ONE_FRAME,
    ("action", "executorch"): "executorch's convolution takes 3-d or 4-d input",
    ("superres", "tensorflow"): "onnx2tf slices a shape with a negative size",
    ("superres", "tflite"): "onnx2tf slices a shape with a negative size",
    ("optical_flow", "ncnn"): NCNN_ONE_FRAME,
    ("optical_flow", "tensorflow"): "the onnx2tf raft flow is tens of pixels off",
    ("optical_flow", "tflite"): "the onnx2tf raft flow is tens of pixels off",
    ("sam", "ncnn"): "ncnn cannot permute a 5-rank tensor",
    ("sam", "tensorflow"): "onnx2tf tiles a 4-d tensor with one multiple",
    ("sam", "tflite"): "onnx2tf tiles a 4-d tensor with one multiple",
    ("sam", "migraphx"): "migraphx resizes only nearest and linear",
    ("zero_shot", "tinygrad"): "tinygrad's Gather rejects 2-d constant indices",
    ("zero_shot", "tensorflow"): OWLV2_MEMORY,
    ("zero_shot", "tflite"): OWLV2_MEMORY,
    ("zero_shot", "ncnn"): OWLV2_MEMORY,
    ("zero_shot", "migraphx"): OWLV2_MEMORY,
}
RUNS = "yes"
NOT_PORTED = "no"
TABLE_ENGINES = ("pytorch",) + EXPORTED_MODEL_ENGINES + ("jax", "llamacpp", "mlx")
README_TABLE_START = "<!-- support matrix start -->"
README_TABLE_END = "<!-- support matrix end -->"


def refusal(task, engine, model_name=None):
    return REFUSALS.get((task, engine, model_name)) or REFUSALS.get((task, engine))


def task_refusals(task, engine):
    return [
        (key[2] if len(key) > 2 else None, reason)
        for key, reason in REFUSALS.items()
        if key[:2] == (task, engine)
    ]


# a cell reads yes, no, or no or partial with the numbers of its footnotes
def markdown_table():
    footnotes = []

    def footnote(model_name, reason):
        note = reason if model_name is None else f"{model_name}: {reason}"
        if note not in footnotes:
            footnotes.append(note)
        return footnotes.index(note) + 1

    def cell(task, engine):
        if engine == "pytorch":
            return RUNS
        if engine not in TASK_ENGINES[task]:
            return NOT_PORTED
        refusals = task_refusals(task, engine)
        if not refusals:
            return RUNS
        numbers = ", ".join(str(footnote(*entry)) for entry in refusals)
        whole_task = any(model_name is None for model_name, _ in refusals)
        return f"{NOT_PORTED if whole_task else 'partial'} [{numbers}]"

    header = "| task | " + " | ".join(TABLE_ENGINES) + " |"
    rule = "|" + "---|" * (len(TABLE_ENGINES) + 1)
    rows = [
        f"| {task} | "
        + " | ".join(cell(task, engine) for engine in TABLE_ENGINES)
        + " |"
        for task in TASK_ENGINES
    ]
    notes = [f"{number}. {note}" for number, note in enumerate(footnotes, 1)]
    return "\n".join([header, rule, *rows, "", *notes])


if __name__ == "__main__":
    print(markdown_table())
