#!/usr/bin/env python3
# Football demo model download
# Copyright (C) 2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Download the football detector weights from the Hugging Face Hub into
# models/football/ (gitignored). Usage:
#   python demo/football/fetch_models.py            # pt + fp16 + ball, what run.sh uses
#   python demo/football/fetch_models.py int8 onnx  # named variants
#   python demo/football/fetch_models.py all

import os
import sys

from huggingface_hub import hf_hub_download

REPO_ID = "boxerab/gst-python-ml-football"
LOCAL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "football"
)
# run.sh's BACKEND value -> the file it loads
VARIANTS = {
    "pt": "football.pt",
    "fp16": "football_fp16.onnx",
    "onnx": "football.onnx",
    "int8": "football_int8.onnx",
    "ball": "ball11n.pt",
}


def main(argv):
    wanted = argv[1:] or ["pt", "fp16", "ball"]
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
