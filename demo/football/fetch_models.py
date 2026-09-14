#!/usr/bin/env python3
# Football demo model download
# Copyright (C) 2026 Collabora Ltd.
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
#
# Download the football detector weights from the Hugging Face Hub into
# models/football/ (gitignored). Usage:
#   python demo/football/fetch_models.py            # pt + fp16, what run.sh uses
#   python demo/football/fetch_models.py int8 onnx  # named variants
#   python demo/football/fetch_models.py all

import os
import sys

from huggingface_hub import hf_hub_download

REPO_ID = "collabora/gst-python-ml-football"
LOCAL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "football"
)
# run.sh's BACKEND value -> the file it loads
VARIANTS = {
    "pt": "football.pt",
    "fp16": "football_fp16.onnx",
    "onnx": "football.onnx",
    "int8": "football_int8.onnx",
}


def main(argv):
    wanted = argv[1:] or ["pt", "fp16"]
    if wanted == ["all"]:
        wanted = list(VARIANTS)
    for variant in wanted:
        if variant not in VARIANTS:
            sys.exit(
                f"unknown model variant {variant!r}; "
                f"choose from {', '.join(VARIANTS)} or all"
            )
        local = os.path.join(LOCAL_DIR, VARIANTS[variant])
        if os.path.isfile(local):
            print(local)
            continue
        path = hf_hub_download(
            repo_id=REPO_ID, filename=VARIANTS[variant], local_dir=LOCAL_DIR
        )
        print(path)


if __name__ == "__main__":
    main(sys.argv)
