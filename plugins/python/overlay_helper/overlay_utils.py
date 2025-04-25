# overlay_utils.py
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

import os
import json


def load_metadata(meta_path, logger):
    """Load JSON metadata from a file and return a dictionary indexed by frame index.

    Args:
        meta_path (str): Path to the JSON metadata file.

    Returns:
        dict: Metadata indexed by frame index.
    """
    if not meta_path:
        logger.error("Frame metadata file path not set.")
        return {}

    if not os.path.exists(meta_path):
        logger.error(f"JSON file not found: {meta_path}")
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
            logger.info(f"Loaded metadata for {len(metadata)} frames.")
            return metadata
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON file: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error while loading metadata: {e}")
        return {}
