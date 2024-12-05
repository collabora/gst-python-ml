# ObjectDetector
# Copyright (C) 2024 Collabora Ltd.
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

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import Gst, GObject  # noqa: E402

CAN_REGISTER_ELEMENT = True
try:
    from gst_analytics_object_detector import GstAnalyticsObjectDetector
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    Gst.warning(f"The 'objectdetector' element will not be available. Error: {e}")


class ObjectDetector(GstAnalyticsObjectDetector):
    """
    GStreamer element for a general object detector where the user sets the model-name property.
    """

    __gstmetadata__ = (
        "ObjectDetector",
        "Transform",
        "General purpose object",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    def __init__(self):
        super(ObjectDetector, self).__init__()
        Gst.info(
            "ObjectDetector created without a model. Please set the 'model-name' property."
        )


if CAN_REGISTER_ELEMENT:
    GObject.type_register(ObjectDetector)
    __gstelementfactory__ = ("objectdetector", Gst.Rank.NONE, ObjectDetector)
else:
    Gst.warning(
        "The 'objectdetector' element will not be registered because gst_analytics_object_detector module is missing."
    )
