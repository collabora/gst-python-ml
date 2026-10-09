# ModelEngines
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

import importlib.util
import os
import platform
import re
import sys
from functools import cache, partial
from pathlib import Path
from typing import NamedTuple

from .engine_factory import EngineFactory
from .hub_causal_lm import (
    CAUSAL_LM_SUFFIX,
    HUB_NAME_SEPARATOR,
    WHISPER_ARCHITECTURE,
    hub_architectures,
)
from .ml_engine import is_torchvision_resnet
from .super_res_engine import CHECKPOINT_URLS
from .support_matrix import (
    ANOMALY_TASK,
    CLASSIFIER_TASK,
    DIRECTORY_INPUT,
    ENGINE_DEVICES,
    ENGINE_EXTRAS,
    ENGINE_PACKAGES,
    FILE_ENGINES,
    OPTICAL_FLOW_MODELS,
    OPTICAL_FLOW_TASK,
    PYTORCH_ONLY_ARCHITECTURES,
    SUPERRES_TASK,
    TABLE_ENGINES,
    TASK_ARCHITECTURES,
    TASK_ENGINES,
    ULTRALYTICS_NAME_PATTERN,
    ULTRALYTICS_VARIANT_TASKS,
    UNTESTED_ULTRALYTICS_VARIANTS,
    UNTESTED_VARIANT_NOTE,
    refusal,
    task_refusals,
)

STATUS_RUNS = "runs"
# another model of the task is refused
STATUS_PARTIAL = "partial"
STATUS_REFUSED = "refused"
STATUS_NOT_PORTED = "not ported"
STATUS_PYTORCH_ONLY = "pytorch only"
STATUS_UNTESTED = "untested"
FILE_TASK = "file"
MISSING_FILE_NOTE = "not on disk"
LLM_TASK = "llm"
WHISPER_TASK = "whisper"
CPU_DEVICE = "cpu"
OPENVINO_GPU = "gpu"
OPENVINO_NPU = "npu"
OPENVINO_DEVICE_SEPARATOR = "."
JAX_GPU_PLATFORM = "gpu"
JAX_TPU_PLATFORM = "tpu"
APPLE_PLATFORM = "darwin"
APPLE_SILICON_MACHINE = "arm64"
PYTORCH_ENGINE = EngineFactory.PYTORCH_ENGINE
ENGINE_ORDER = TABLE_ENGINES + tuple(
    engine for engine in ENGINE_PACKAGES if engine not in TABLE_ENGINES
)


class ResolvedTask(NamedTuple):
    task: str
    note: str = ""
    file_kind: str | None = None
    pytorch_only: bool = False


def is_installed(engine):
    # a dotted name raises when its parent package is missing
    try:
        return importlib.util.find_spec(ENGINE_PACKAGES[engine]) is not None
    except ModuleNotFoundError:
        return False


def is_torchvision_classifier(model_name):
    from torchvision import models

    return model_name in models.list_models(module=models)


def model_file_kind(model):
    if os.path.isdir(model):
        return DIRECTORY_INPUT
    suffix = Path(model).suffix
    return suffix if suffix in FILE_ENGINES else None


def file_task(model, file_kind):
    notes = [file_kind] if os.path.exists(model) else [file_kind, MISSING_FILE_NOTE]
    return ResolvedTask(FILE_TASK, ", ".join(notes), file_kind=file_kind)


def architecture_tasks(architecture):
    if architecture.endswith(CAUSAL_LM_SUFFIX):
        return [ResolvedTask(LLM_TASK)]
    if architecture == WHISPER_ARCHITECTURE:
        return [ResolvedTask(WHISPER_TASK)]
    note = PYTORCH_ONLY_ARCHITECTURES.get(architecture, "")
    return [
        ResolvedTask(task, note, pytorch_only=bool(note))
        for task in TASK_ARCHITECTURES.get(architecture, ())
    ]


def name_tasks(model_name):
    ultralytics = re.fullmatch(ULTRALYTICS_NAME_PATTERN, model_name)
    if ultralytics:
        variant = ultralytics["variant"]
        if variant not in ULTRALYTICS_VARIANT_TASKS:
            return []
        untested = variant in UNTESTED_ULTRALYTICS_VARIANTS
        note = UNTESTED_VARIANT_NOTE if untested else ""
        return [ResolvedTask(ULTRALYTICS_VARIANT_TASKS[variant], note)]
    if model_name in OPTICAL_FLOW_MODELS:
        return [ResolvedTask(OPTICAL_FLOW_TASK)]
    if model_name in CHECKPOINT_URLS:
        return [ResolvedTask(SUPERRES_TASK)]
    if not is_torchvision_classifier(model_name):
        return []
    tasks = [ResolvedTask(CLASSIFIER_TASK)]
    if is_torchvision_resnet(model_name):
        tasks.append(ResolvedTask(ANOMALY_TASK))
    return tasks


def resolve(model):
    file_kind = model_file_kind(model)
    if file_kind:
        return [file_task(model, file_kind)]
    if HUB_NAME_SEPARATOR in model:
        return [
            resolved
            for architecture in hub_architectures(model)
            for resolved in architecture_tasks(architecture)
        ]
    return name_tasks(model)


def task_status(task, engine, model):
    if engine == PYTORCH_ENGINE:
        return STATUS_RUNS, ""
    if engine not in TASK_ENGINES[task]:
        return STATUS_NOT_PORTED, ""
    reason = refusal(task, engine, model)
    if reason:
        return STATUS_REFUSED, reason
    other_models = task_refusals(task, engine)
    if other_models:
        reasons = [f"{name}: {reason}" for name, reason in other_models]
        return STATUS_PARTIAL, ", ".join(reasons)
    return STATUS_RUNS, ""


def engine_statuses(model, resolved):
    if resolved.file_kind:
        engines = FILE_ENGINES[resolved.file_kind]
        return [
            (engine, STATUS_UNTESTED, "")
            for engine in ENGINE_ORDER
            if engine in engines
        ]
    if resolved.pytorch_only or resolved.task not in TASK_ENGINES:
        return [(PYTORCH_ENGINE, STATUS_PYTORCH_ONLY, "")]
    return [
        (engine, *task_status(resolved.task, engine, model)) for engine in TABLE_ENGINES
    ]


def engine_entry(engine, status, reason):
    return {
        "engine": engine,
        "status": status,
        "reason": reason,
        "devices": list(ENGINE_DEVICES[engine]),
        "installed": is_installed(engine),
        "extra": list(ENGINE_EXTRAS[engine]),
    }


def engine_options(model):
    return {
        "model": model,
        "tasks": [
            {
                "task": resolved.task,
                "note": resolved.note,
                "engines": [
                    engine_entry(*engine_status)
                    for engine_status in engine_statuses(model, resolved)
                ],
            }
            for resolved in resolve(model)
        ],
    }


@cache
def torch_sees_gpu():
    import torch

    return torch.cuda.is_available()


def nvidia_gpu():
    import torch

    return torch_sees_gpu() and torch.version.hip is None


def amd_gpu():
    import torch

    return torch_sees_gpu() and torch.version.hip is not None


@cache
def openvino_device_kinds():
    import openvino

    return {
        name.split(OPENVINO_DEVICE_SEPARATOR)[0].lower()
        for name in openvino.Core().available_devices
    }


def openvino_has_device(device):
    return device in openvino_device_kinds()


def vulkan_gpu():
    import ncnn

    return ncnn.get_gpu_count() > 0


def apple_silicon():
    return (
        sys.platform == APPLE_PLATFORM and platform.machine() == APPLE_SILICON_MACHINE
    )


# jax raises when it has no backend for the platform
def jax_has_platform(platform_name):
    import jax

    return bool(jax.devices(platform_name))


# a device missing here is not probed
DEVICE_PROBES = {
    ("pytorch", "cuda"): torch_sees_gpu,
    ("onnx", "cuda"): nvidia_gpu,
    ("onnx", "tensorrt"): nvidia_gpu,
    ("onnx", "rocm"): amd_gpu,
    ("onnx", "migraphx"): amd_gpu,
    ("onnx", "coreml"): apple_silicon,
    ("openvino", OPENVINO_GPU): partial(openvino_has_device, OPENVINO_GPU),
    ("openvino", OPENVINO_NPU): partial(openvino_has_device, OPENVINO_NPU),
    ("tvm", "cuda"): nvidia_gpu,
    ("tensorflow", "cuda"): nvidia_gpu,
    ("ncnn", "vulkan"): vulkan_gpu,
    ("executorch", "coreml"): apple_silicon,
    ("executorch", "mps"): apple_silicon,
    ("iree", "cuda"): nvidia_gpu,
    ("iree", "rocm"): amd_gpu,
    ("iree", "vulkan"): vulkan_gpu,
    ("iree", "metal"): apple_silicon,
    ("tinygrad", "cuda"): nvidia_gpu,
    ("migraphx", "gpu"): amd_gpu,
    ("jax", JAX_GPU_PLATFORM): partial(jax_has_platform, JAX_GPU_PLATFORM),
    ("jax", JAX_TPU_PLATFORM): partial(jax_has_platform, JAX_TPU_PLATFORM),
    ("llamacpp", "cuda"): nvidia_gpu,
    ("llamacpp", "metal"): apple_silicon,
    ("mlx", "gpu"): apple_silicon,
}


def device_reachable(engine, device):
    if device == CPU_DEVICE:
        return True
    probe = DEVICE_PROBES.get((engine, device))
    if probe is None:
        return False
    # a missing package or a crashing driver leaves the device unreachable
    try:
        return probe()
    except Exception:
        return False


def unprobed_devices(engine):
    return tuple(
        device
        for device in ENGINE_DEVICES[engine]
        if device != CPU_DEVICE and (engine, device) not in DEVICE_PROBES
    )


def available_devices():
    return {
        engine: tuple(
            device
            for device in ENGINE_DEVICES[engine]
            if device_reachable(engine, device)
        )
        for engine in ENGINE_ORDER
        if is_installed(engine)
    }


def with_available_devices(options):
    return {**options, "available_devices": available_devices()}
