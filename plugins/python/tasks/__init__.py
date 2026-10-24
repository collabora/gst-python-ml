# Portable ML task logic
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Backend-agnostic ML task logic.

Each module here holds the inference and result-handling steps for one kind of
element (object detection, ...), expressed only in terms of the ML engine,
numpy results, and the analytics facade (`backend.analytics`). A backend
combines a task mixin with a framework element shell, which supplies the
per-buffer glue (do_transform_ip / process, pad templates, buffer I/O).
"""
