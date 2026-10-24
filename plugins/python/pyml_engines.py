# pyml-engines
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
import sys

from engine.benchmark import benchmark_rows, status_text
from engine.model_engines import (
    available_devices,
    engine_options,
    with_available_devices,
    unprobed_devices,
)

REACHABLE_MARK = "*"
UNPROBED_MARK = "?"
LEGEND = f"{REACHABLE_MARK} reachable on this machine, {UNPROBED_MARK} not probed"
PACKAGE_NAME = "modelpipe"
NO_PIP_EXTRA = "not on PyPI"
DEFAULT_BENCHMARK_FRAMES = 20
BENCHMARK_HEADER = ("task", "engine", "device", "load s", "ms/frame", "fps", "note")


def marked_devices(engine, devices, reachable):
    unprobed = unprobed_devices(engine)
    marks = []
    for device in devices:
        mark = REACHABLE_MARK if device in reachable else ""
        if device in unprobed:
            mark = UNPROBED_MARK
        marks.append(device + mark)
    return " ".join(marks)


def install_hint(entry):
    if entry["installed"]:
        return ""
    if not entry["extra"]:
        return NO_PIP_EXTRA
    first_extra, *other_extras = entry["extra"]
    alternatives = "".join(f" or [{extra}]" for extra in other_extras)
    return f"pip install {PACKAGE_NAME}[{first_extra}]{alternatives}"


def task_lines(task, reachable_devices):
    heading = f"{task['task']}: {task['note']}" if task["note"] else task["task"]
    rows = [
        (
            entry["engine"],
            status_text(entry),
            marked_devices(
                entry["engine"],
                entry["devices"],
                reachable_devices.get(entry["engine"], ()),
            ),
            install_hint(entry),
        )
        for entry in task["engines"]
    ]
    return [heading, *(f"  {line}" for line in aligned_lines(rows))]


def aligned_lines(rows):
    aligned_columns = len(rows[0]) - 1
    widths = [
        max(len(row[column]) for row in rows) for column in range(aligned_columns)
    ]
    return [
        "  ".join(
            [*(cell.ljust(width) for cell, width in zip(row, widths)), row[-1]]
        ).rstrip()
        for row in rows
    ]


def one_decimal(value):
    return "" if value is None else f"{value:.1f}"


def benchmark_lines(rows):
    cells = [
        (
            row["task"],
            row["engine"],
            row["device"],
            one_decimal(row["load_seconds"]),
            one_decimal(row["latency_ms"]),
            one_decimal(row["fps"]),
            row["note"],
        )
        for row in rows
    ]
    return aligned_lines([BENCHMARK_HEADER, *cells])


def print_progress(task, engine, device):
    print(f"{task} {engine} {device}", file=sys.stderr, flush=True)


def print_benchmark(options, frames, as_json):
    rows = benchmark_rows(options, frames, print_progress)
    if as_json:
        report = {"model": options["model"], "frames": frames, "results": rows}
        print(json.dumps(report, indent=2))
        return
    print("\n".join(benchmark_lines(rows)))


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="pyml-engines",
        description="List the engines and devices that run a model name or file.",
    )
    parser.add_argument("model", help="a Hugging Face id, a model name or a file")
    parser.add_argument("--json", action="store_true", help="print JSON")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="time each engine and device that runs the model here",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_BENCHMARK_FRAMES,
        help="frames to time per engine and device",
    )
    arguments = parser.parse_args(argv)
    if arguments.frames < 1:
        parser.error("--frames must be at least 1")
    try:
        options = engine_options(arguments.model)
    except OSError as error:
        first_line = str(error).splitlines()[0]
        sys.exit(f"pyml-engines: cannot read {arguments.model}: {first_line}")
    if not options["tasks"]:
        sys.exit(f"pyml-engines: no task or engine matches {arguments.model}")
    if arguments.benchmark:
        print_benchmark(options, arguments.frames, arguments.json)
        return
    if arguments.json:
        print(json.dumps(with_available_devices(options), indent=2))
        return
    reachable_devices = available_devices()
    blocks = [
        "\n".join(task_lines(task, reachable_devices)) for task in options["tasks"]
    ]
    print("\n\n".join([*blocks, LEGEND]))


if __name__ == "__main__":
    main()
