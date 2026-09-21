# documented_pipelines
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

import re
from pathlib import Path

LAUNCHER_PREFIX = "python pyml-launch.py"

PIPELINE_PATTERN = re.compile(
    r"(?:`)?\s*(python pyml-launch\.py\s+.*?)(?:`)?(?=\n\n|\n\s*\n|$)", re.DOTALL
)
HEADING_PATTERN = re.compile(r"^#{2,3} (.+)$", re.MULTILINE)


def preceding_heading(headings, position):
    heading = ""
    for start, text in headings:
        if start > position:
            break
        heading = text
    return heading


def pipelines_by_section(doc_path):
    content = Path(doc_path).read_text()
    headings = [
        (match.start(), match.group(1).strip())
        for match in HEADING_PATTERN.finditer(content)
    ]
    sections = {}
    for match in PIPELINE_PATTERN.finditer(content):
        description = match.group(1).strip().strip("`")
        description = description.removeprefix(LAUNCHER_PREFIX).strip()
        heading = preceding_heading(headings, match.start())
        sections.setdefault(heading, []).append(description)
    return sections
