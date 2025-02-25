#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path

def create_symlink_structure(source_dir, target_dir):
    """
    Recursively create a directory structure with symlinks for all files and directories.

    Args:
        source_dir (Path): The original directory to replicate.
        target_dir (Path): The destination directory for symlinks.
    """

    if not source_dir.exists() or not source_dir.is_dir():
        print(f"ERROR: Source directory '{source_dir}' does not exist or is not a directory.")
        sys.exit(1)
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)

    for root, dirs, files in os.walk(source_dir):
        source_root = Path(root)
        relative_path = source_root.relative_to(source_dir)
        target_root = target_dir / relative_path
       
        # Create symlinks for all directories
        if not target_root.exists():
            target_root.mkdir(parents=True, exist_ok=True)

        for dir_name in dirs:
            source_dir_path = source_root / dir_name
            target_dir_path = target_root / dir_name
            if not target_dir_path.exists():
                try:
                    target_dir_path.symlink_to(source_dir_path, target_is_directory=True)
                except OSError as e:
                    print(f"ERROR: Failed to create directory symlink '{target_dir_path}': {e}")

        # Create symlinks for all files
        for file_name in files:
            source_file_path = source_root / file_name
            target_file_path = target_root / file_name
            if not target_file_path.exists():
                try:
                    target_file_path.symlink_to(source_file_path)
                except OSError as e:
                    print(f"ERROR: Failed to create file symlink '{target_file_path}': {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Create a directory structure with symlinks for all files and directories."
    )
    parser.add_argument("source", type=str, help="Path to the source directory")
    parser.add_argument("target", type=str, help="Path to the target directory for symlinks")
    args = parser.parse_args()

    source_dir = Path(args.source).resolve()
    target_dir = Path(args.target).resolve()

    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"WARNING: Target directory '{target_dir}' exists and is not empty.")
        confirm = input("Proceed? (y/N): ")
        if confirm.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    create_symlink_structure(source_dir, target_dir)

if __name__ == "__main__":
    main()