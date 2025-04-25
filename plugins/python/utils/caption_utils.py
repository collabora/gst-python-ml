# caption_utils.py
# Copyright (C) 2024-2025 Collabora Ltd.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Library General Public License for more details.
#
# You should have received a copy of the GNU Library General Public
# License along with this library; if not, write to the
# Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
# Boston, MA 02110-1301, USA.

import os
import json


def load_captions(caption_path, logger):
    """
    Load JSON caption data from a file and return a dictionary indexed by frame index.

    Args:
        caption_path (str): Path to the JSON caption file.
        logger: Logger instance for logging messages.

    Returns:
        dict: Captions indexed by frame index (e.g., {0: "caption text", 1: "other text"}).
    """
    if not caption_path:
        logger.error("Caption file path not set.")
        return {}

    if not os.path.exists(caption_path):
        logger.error(f"JSON file not found: {caption_path}")
        return {}

    try:
        with open(caption_path, "r") as f:
            all_data = json.load(f)
            frame_data = all_data.get("frames", [])
            # Store captions indexed by frame_index
            captions = {
                frame.get("frame_index"): frame.get("caption", "")
                for frame in frame_data
            }
            logger.info(f"Loaded captions for {len(captions)} frames.")
            return captions
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON file: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error while loading captions: {e}")
        return {}
