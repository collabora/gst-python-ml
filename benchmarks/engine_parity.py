# Engine parity benchmark
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

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_DIRECTORY = REPOSITORY_ROOT / "plugins" / "python"

# the plugin loader reads GST_PLUGIN_PATH at Gst.init
os.environ["GST_PLUGIN_PATH"] = str(REPOSITORY_ROOT / "plugins")
sys.path.insert(0, str(PLUGIN_DIRECTORY))

import gi  # noqa: E402

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

Gst.init(None)

from engine.engine_factory import EngineFactory  # noqa: E402
from utils.box_matching import matched_box_fraction  # noqa: E402

VIDEO = REPOSITORY_ROOT / "data" / "people.mp4"
DEFAULT_ENGINE_MODELS = ["pytorch=yolo11m", "onnx=yolo11m-384x640.onnx"]
DEFAULT_FRAMES = 100
DEFAULT_DEVICE = "cuda"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
IOU_THRESHOLD = 0.5
REFERENCE_ENGINE = "pytorch"
PIPELINE_TIMEOUT = 600 * Gst.SECOND
# the first frames pay for a jit or a lazy cuda start
WARMUP_FRAMES = 10
# pts of the same decoded frame is identical across runs, rounding only guards float text
PTS_DECIMALS = 6

# the two elements default to different thresholds
THRESHOLDS = "confidence=0.25 nms-iou=0.45"
PYTORCH_DETECTOR = f"pyml_yolo model-name={{model}} device={{device}} {THRESHOLDS}"
CONVERTED_MODEL_DETECTOR = (
    "pyml_objectdetector engine-name={engine} model-name={model} device={device} "
    f"input-format=nchw post-process=anchor_free {THRESHOLDS}"
)

COLUMN_HEADINGS = [
    "engine",
    "device",
    "model",
    "frames",
    "fps",
    "detections/frame",
    "mean score",
    f"matched at IoU {IOU_THRESHOLD}",
]


def detector_description(engine, model, device):
    template = (
        PYTORCH_DETECTOR if engine == REFERENCE_ENGINE else CONVERTED_MODEL_DETECTOR
    )
    return template.format(engine=engine, model=model, device=device)


def model_argument(model):
    beside_the_repository = REPOSITORY_ROOT / model
    return str(beside_the_repository) if beside_the_repository.is_file() else model


def run_pipeline(detector, frames, records_path):
    pipeline = Gst.parse_launch(
        f"filesrc location={VIDEO} ! decodebin ! videoconvert ! videoscale "
        f"! video/x-raw,width={FRAME_WIDTH},height={FRAME_HEIGHT} "
        f"! identity name=limiter eos-after={frames} "
        f"! {detector} ! pyml_metasink location={records_path}"
    )
    arrivals = []

    # identity pushes a few buffers past eos-after
    def count_frame(pad, info):
        if len(arrivals) >= frames:
            return Gst.PadProbeReturn.DROP
        arrivals.append(time.perf_counter())
        return Gst.PadProbeReturn.OK

    limiter = pipeline.get_by_name("limiter")
    limiter.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, count_frame)

    pipeline.set_state(Gst.State.PLAYING)
    message = pipeline.get_bus().timed_pop_filtered(
        PIPELINE_TIMEOUT, Gst.MessageType.EOS | Gst.MessageType.ERROR
    )
    finished = time.perf_counter()
    pipeline.set_state(Gst.State.NULL)
    if message is None:
        raise RuntimeError("the pipeline did not finish")
    if message.type == Gst.MessageType.ERROR:
        error, debug = message.parse_error()
        raise RuntimeError(f"{error.message}\n{debug}")
    if len(arrivals) <= WARMUP_FRAMES:
        raise RuntimeError(f"only {len(arrivals)} frames reached the detector")
    timed_frames = len(arrivals) - WARMUP_FRAMES
    return timed_frames / (finished - arrivals[WARMUP_FRAMES]), len(arrivals)


def boxes_by_time(records_path):
    boxes = {}
    for line in records_path.read_text().splitlines():
        record = json.loads(line)
        detections = record.get("detections")
        if not detections:
            continue
        boxes[round(record["pts"], PTS_DECIMALS)] = detections
    return boxes


def measure(engine, model, device, frames, records_path):
    detector = detector_description(engine, model_argument(model), device)
    print(f"running {detector}", file=sys.stderr)
    frames_per_second, frames_seen = run_pipeline(detector, frames, records_path)
    boxes = boxes_by_time(records_path)
    detections = [box for frame_boxes in boxes.values() for box in frame_boxes]
    scores = [box["score"] for box in detections]
    return {
        "engine": engine,
        "device": device,
        "model": model,
        "frames": frames_seen,
        "fps": frames_per_second,
        "detections_per_frame": len(detections) / frames_seen if frames_seen else 0.0,
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
        "boxes": boxes,
    }


def markdown_table(rows):
    lines = [
        "| " + " | ".join(COLUMN_HEADINGS) + " |",
        "| " + " | ".join("---" for _ in COLUMN_HEADINGS) + " |",
    ]
    for row in rows:
        matched = row["matched"]
        lines.append(
            "| {engine} | {device} | {model} | {frames} | {fps:.1f} | {detections:.2f} "
            "| {score:.3f} | {matched} |".format(
                engine=row["engine"],
                device=row["device"],
                model=row["model"],
                frames=row["frames"],
                fps=row["fps"],
                detections=row["detections_per_frame"],
                score=row["mean_score"],
                matched="n/a" if matched is None else f"{matched:.2f}",
            )
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Run the same detection over every engine and print a table"
    )
    parser.add_argument("engine_models", nargs="*", metavar="engine=model[@device]")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    arguments = parser.parse_args()

    rows = []
    with tempfile.TemporaryDirectory() as directory:
        pairs = arguments.engine_models or DEFAULT_ENGINE_MODELS
        for index, pair in enumerate(pairs):
            engine, _, model_and_device = pair.partition("=")
            model, _, device = model_and_device.partition("@")
            if not engine or not model:
                parser.error(f"{pair!r} is not an engine=model pair")
            records_path = Path(directory) / f"{index}-{engine}.jsonl"
            try:
                EngineFactory.create(engine)
                rows.append(
                    measure(
                        engine,
                        model,
                        device or arguments.device,
                        arguments.frames,
                        records_path,
                    )
                )
            except Exception as error:
                print(f"skipping {pair}: {error}", file=sys.stderr)

    reference = next(
        (row["boxes"] for row in rows if row["engine"] == REFERENCE_ENGINE), None
    )
    for row in rows:
        row["matched"] = (
            None
            if reference is None
            else matched_box_fraction(reference, row["boxes"], IOU_THRESHOLD)
        )
    print(markdown_table(rows))


if __name__ == "__main__":
    main()
