# GstStableDiffusion
# Copyright (C) 2024-2025 Collabora Ltd.
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

import subprocess
import os
import json
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402


def runtime_check_gstreamer_version(min_version="1.24"):
    try:
        result = subprocess.run(
            ["gst-launch-1.0", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            version_line = result.stdout.splitlines()[0]
            current_version = version_line.split(" ")[-1]
            if current_version >= min_version:
                return True
            else:
                raise EnvironmentError(
                    f"GStreamer version {min_version} or higher is required, but version {current_version} is installed."
                )
        else:
            raise EnvironmentError(
                "GStreamer is not installed or not found in the PATH."
            )
    except FileNotFoundError:
        raise EnvironmentError("GStreamer is not installed.")


def load_metadata(meta_path):
    """Load JSON metadata from a file and return a dictionary indexed by frame index.

    Args:
        meta_path (str): Path to the JSON metadata file.

    Returns:
        dict: Metadata indexed by frame index.
    """
    if not meta_path:
        Gst.error("Frame metadata file path not set.")
        return {}

    if not os.path.exists(meta_path):
        Gst.error(f"JSON file not found: {meta_path}")
        return {}

    try:
        with open(meta_path, "r") as f:
            all_data = json.load(f)
            frame_data = all_data.get("frames", [])
            # Store metadata indexed by frame_index
            metadata = {
                frame.get("frame_index"): frame.get("objects", [])
                for frame in frame_data
            }
            Gst.info(f"Loaded metadata for {len(metadata)} frames.")
            return metadata
    except json.JSONDecodeError as e:
        Gst.error(f"Failed to parse JSON file: {e}")
        return {}
    except Exception as e:
        Gst.error(f"Unexpected error while loading metadata: {e}")
        return {}
