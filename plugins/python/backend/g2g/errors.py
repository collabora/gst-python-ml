# Element error reporting (g2g backend)
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.


# the gst counterpart posts on the element's bus, a g2g host only sees the exception
def post_error(element, summary, exception):
    raise exception


def post_model_load_error(element, model_name, exception):
    raise exception
