# BaseTransform
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Compatibility shim. `BaseTransform` now lives behind the pluggable backend
in `backend/` (see `backend/__init__.py`). Leaf plugins keep importing it from
here; the active backend is chosen by the MP_BACKEND environment variable.
"""

from backend import BaseTransform

__all__ = ["BaseTransform"]
