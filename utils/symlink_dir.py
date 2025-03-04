# symlink_dir
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


#!/usr/bin/env python3

"""
A script to recursively create a directory structure with symbolic links
mirroring a source directory's files and folders.

Usage example:
    python3 symlink_dir.py /path/to/source /path/to/target
"""

import os  # For directory walking and filesystem operations
import sys  # For system exit and error handling
import argparse  # For parsing command-line arguments
from pathlib import Path  # For cross-platform path handling


def create_symlink_structure(source_dir, target_dir):
    """
    Recursively create a directory structure with symlinks for all files and directories.

    This function walks through the source directory and creates a matching structure
    in the target directory using symbolic links instead of copying files. It preserves
    the original directory hierarchy and creates symlinks for both files and directories.

    Args:
        source_dir (Path): The original directory to replicate
        target_dir (Path): The destination directory where symlinks will be created

    Raises:
        SystemExit: If source directory doesn't exist or isn't a directory
        OSError: If symlink creation fails (printed as warning but continues execution)
    """
    # Validate source directory
    if not source_dir.exists() or not source_dir.is_dir():
        print(
            f"ERROR: Source directory '{source_dir}' does not exist or is not a directory."
        )
        sys.exit(1)

    # Create target directory if it doesn't exist
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)

    # Walk through source directory structure
    for root, dirs, files in os.walk(source_dir):
        source_root = Path(root)
        # Calculate relative path for maintaining structure
        relative_path = source_root.relative_to(source_dir)
        target_root = target_dir / relative_path

        # Create target directory if it doesn't exist
        if not target_root.exists():
            target_root.mkdir(parents=True, exist_ok=True)

        # Create symlinks for all subdirectories
        for dir_name in dirs:
            source_dir_path = source_root / dir_name
            target_dir_path = target_root / dir_name
            if not target_dir_path.exists():
                try:
                    target_dir_path.symlink_to(
                        source_dir_path, target_is_directory=True
                    )
                except OSError as e:
                    print(
                        f"ERROR: Failed to create directory symlink '{target_dir_path}': {e}"
                    )

        # Create symlinks for all files
        for file_name in files:
            source_file_path = source_root / file_name
            target_file_path = target_root / file_name
            if not target_file_path.exists():
                try:
                    target_file_path.symlink_to(source_file_path)
                except OSError as e:
                    print(
                        f"ERROR: Failed to create file symlink '{target_file_path}': {e}"
                    )


def main():
    """
    Main function to parse arguments and execute the symlink structure creation.

    Handles command-line arguments and user confirmation if target directory exists.
    Calls create_symlink_structure with validated paths.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Create a directory structure with symlinks for all files and directories."
    )
    parser.add_argument("source", type=str, help="Path to the source directory")
    parser.add_argument(
        "target", type=str, help="Path to the target directory for symlinks"
    )
    args = parser.parse_args()

    # Convert to absolute paths
    source_dir = Path(args.source).resolve()
    target_dir = Path(args.target).resolve()

    # Warn if target directory exists and isn't empty
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"WARNING: Target directory '{target_dir}' exists and is not empty.")
        confirm = input("Proceed? (y/N): ")
        if confirm.lower() != "y":
            print("Aborted.")
            sys.exit(0)

    # Execute the symlink creation
    create_symlink_structure(source_dir, target_dir)


if __name__ == "__main__":
    main()
