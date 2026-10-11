# Benchmark
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

import importlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, NamedTuple

from . import model_engines
from .model_engines import STATUS_PARTIAL, STATUS_PYTORCH_ONLY, STATUS_RUNS

BENCHMARKED_STATUSES = (STATUS_RUNS, STATUS_PARTIAL, STATUS_PYTORCH_ONLY)
NOT_INSTALLED_NOTE = "not installed"
NO_DRIVER_NOTE = "no benchmark driver"
DEFAULT_FRAME_WIDTH = 640
DEFAULT_FRAME_HEIGHT = 480
COLOR_CHANNELS = 3
PIXEL_VALUE_LIMIT = 256
RANDOM_SEED = 0
ACTION_WINDOW_FRAMES = 16
FLOW_SHIFT_PIXELS = 8
CLIP_LABELS = ("a person", "a car", "a dog", "a tree")
ZERO_SHOT_LABELS = "a face, a blue chair, a dog"
# a 2x upscale of 640x480 on the cpu is too slow for a sweep
SUPERRES_FRAME_WIDTH = 320
SUPERRES_FRAME_HEIGHT = 240
# the size the exported graph takes
FLOW_FRAME_WIDTH = 640
FLOW_FRAME_HEIGHT = 360
MILLISECONDS_PER_SECOND = 1000
# a jit engine captures its kernels on the second call
WARMUP_FRAMES = 2
PLUGIN_DIRECTORY = Path(__file__).resolve().parent.parent


def call_forward(element, frame):
    return element.forward(frame)


def call_do_forward(element, frame):
    return element.do_forward(frame)


def call_clip_engine(element, frame):
    return element.task_engine.do_forward(frame)


def call_action_window(element, frame):
    return element.forward([frame] * ACTION_WINDOW_FRAMES)


def call_flow_pair(element, frame):
    import numpy as np

    return element.forward(frame, np.roll(frame, FLOW_SHIFT_PIXELS, axis=1))


class BenchmarkDriver(NamedTuple):
    module: str
    element_class: str
    forward: Callable[[Any, Any], Any]
    frame_width: int = DEFAULT_FRAME_WIDTH
    frame_height: int = DEFAULT_FRAME_HEIGHT
    labels_property: str = ""
    clip_labels: tuple[str, ...] = ()


BENCHMARK_DRIVERS = {
    "depth": BenchmarkDriver("depth", "DepthTransform", call_forward),
    "yolo": BenchmarkDriver("yolo", "YOLOTransform", call_do_forward),
    "pose": BenchmarkDriver("pose", "YOLOPoseTransform", call_do_forward),
    "clip": BenchmarkDriver(
        "clip", "CLIPTransform", call_clip_engine, clip_labels=CLIP_LABELS
    ),
    "anomaly": BenchmarkDriver("anomaly", "AnomalyTransform", call_forward),
    "classifier": BenchmarkDriver("classifier", "Classifier", call_do_forward),
    "embedding": BenchmarkDriver("embedding", "EmbeddingTransform", call_forward),
    "action": BenchmarkDriver("action", "ActionTransform", call_action_window),
    "superres": BenchmarkDriver(
        "superres",
        "SuperResTransform",
        call_forward,
        SUPERRES_FRAME_WIDTH,
        SUPERRES_FRAME_HEIGHT,
    ),
    "optical_flow": BenchmarkDriver(
        "optical_flow",
        "OpticalFlowTransform",
        call_flow_pair,
        FLOW_FRAME_WIDTH,
        FLOW_FRAME_HEIGHT,
    ),
    "sam": BenchmarkDriver("sam", "SamTransform", call_forward),
    "zero_shot": BenchmarkDriver(
        "zeroshotdetector",
        "ZeroShotDetector",
        call_do_forward,
        labels_property=ZERO_SHOT_LABELS,
    ),
}


def status_text(entry):
    if not entry["reason"]:
        return entry["status"]
    return f"{entry['status']}: {entry['reason']}"


def loaded_element(driver, model, engine, device):
    element_class = getattr(
        importlib.import_module(driver.module), driver.element_class
    )
    element = element_class()
    if driver.labels_property:
        element.set_property("labels", driver.labels_property)
    element.set_property("model-name", model)
    element.set_property("engine-name", engine)
    element.set_property("device", device)
    element.do_load_model()
    if driver.clip_labels:
        element.task_engine.clip_labels = list(driver.clip_labels)
    return element


def measure(task, model, engine, device, frames):
    import gi
    import numpy as np

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    driver = BENCHMARK_DRIVERS[task]
    load_start = time.perf_counter()
    element = loaded_element(driver, model, engine, device)
    load_seconds = time.perf_counter() - load_start
    frame_shape = (driver.frame_height, driver.frame_width, COLOR_CHANNELS)
    frame = np.random.default_rng(RANDOM_SEED).integers(
        0, PIXEL_VALUE_LIMIT, frame_shape, dtype=np.uint8
    )
    for _ in range(WARMUP_FRAMES):
        driver.forward(element, frame)
    run_start = time.perf_counter()
    for _ in range(frames):
        driver.forward(element, frame)
    run_seconds = time.perf_counter() - run_start
    return {
        "load_seconds": load_seconds,
        "latency_ms": run_seconds * MILLISECONDS_PER_SECOND / frames,
        "fps": frames / run_seconds,
    }


def last_json_object(stdout):
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        return None
    try:
        parsed = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def crash_note(completed):
    if completed.returncode < 0:
        return f"killed by {signal.Signals(-completed.returncode).name}"
    error_lines = [line.strip() for line in completed.stderr.splitlines()]
    error_lines = [line for line in error_lines if line]
    if not error_lines:
        return f"exit code {completed.returncode}"
    return f"exit code {completed.returncode}: {error_lines[-1]}"


def child_result(completed):
    result = last_json_object(completed.stdout) if completed.returncode == 0 else None
    if result is None:
        return {"error": crash_note(completed)}
    return result


# tinygrad and migraphx crash the interpreter on some graphs
def benchmark_in_child(task, model, engine, device, frames):
    python_path = [str(PLUGIN_DIRECTORY), os.environ.get("PYTHONPATH", "")]
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, python_path)),
    }
    command = [sys.executable, "-m", __name__, task, model, engine, device]
    completed = subprocess.run(
        [*command, str(frames)], capture_output=True, text=True, env=environment
    )
    return child_result(completed)


def unmeasured_row(task, engine, note):
    return {
        "task": task,
        "engine": engine,
        "device": "",
        "load_seconds": None,
        "latency_ms": None,
        "fps": None,
        "note": note,
    }


def measured_row(task, engine, device, result):
    if "error" in result:
        return {**unmeasured_row(task, engine, result["error"]), "device": device}
    return {
        "task": task,
        "engine": engine,
        "device": device,
        "load_seconds": result["load_seconds"],
        "latency_ms": result["latency_ms"],
        "fps": result["fps"],
        "note": "",
    }


def engine_rows(task, model, entry, devices, frames, progress):
    engine = entry["engine"]
    if entry["status"] not in BENCHMARKED_STATUSES:
        return [unmeasured_row(task, engine, status_text(entry))]
    if not entry["installed"]:
        return [unmeasured_row(task, engine, NOT_INSTALLED_NOTE)]
    rows = []
    for device in devices.get(engine, ()):
        progress(task, engine, device)
        result = benchmark_in_child(task, model, engine, device, frames)
        rows.append(measured_row(task, engine, device, result))
    return rows


def no_progress(task, engine, device):
    pass


def benchmark_rows(options, frames, progress=no_progress):
    devices = model_engines.available_devices()
    rows = []
    for task in options["tasks"]:
        if task["task"] not in BENCHMARK_DRIVERS:
            rows.append(unmeasured_row(task["task"], "", NO_DRIVER_NOTE))
            continue
        for entry in task["engines"]:
            rows.extend(
                engine_rows(
                    task["task"], options["model"], entry, devices, frames, progress
                )
            )
    return rows


def main(arguments):
    task, model, engine, device, frames = arguments
    try:
        result = measure(task, model, engine, device, int(frames))
    except Exception as error:
        message_lines = str(error).splitlines() or [type(error).__name__]
        result = {"error": message_lines[0]}
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
