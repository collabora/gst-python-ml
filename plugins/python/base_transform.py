# BaseTransform
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

"""Compatibility shim. `BaseTransform` now lives behind the pluggable backend
in `backend/` (see `backend/__init__.py`). Leaf plugins keep importing it from
here; the active backend is chosen by the PYML_BACKEND environment variable.
"""

from backend import BaseTransform

__all__ = ["BaseTransform"]
