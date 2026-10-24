# Element error reporting (g2g backend)
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


# the gst counterpart posts on the element's bus, a g2g host only sees the exception
def post_error(element, summary, exception):
    raise exception


def post_model_load_error(element, model_name, exception):
    raise exception
