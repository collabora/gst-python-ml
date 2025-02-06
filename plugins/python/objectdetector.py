# ObjectDetector
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

CAN_REGISTER_ELEMENT = True
try:
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    gi.require_version("GLib", "2.0")
    from gi.repository import Gst, GObject  # noqa: E402
    from gst_object_detector import GstObjectDetector
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    self.logger.warning(
        f"The 'objectdetector_pylm' element will not be available. Error: {e}"
    )


class ObjectDetector(GstObjectDetector):
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
        super().__init__()
        self.logger.info(
            "ObjectDetector created without a model. Please set the 'model-name' property."
        )


if CAN_REGISTER_ELEMENT:
    GObject.type_register(ObjectDetector)
    __gstelementfactory__ = ("pyml_objectdetector", Gst.Rank.NONE, ObjectDetector)
else:
    self.logger.warning(
        "The 'pyml_objectdetector' element will not be registered because gst_object_detector module is missing."
    )
