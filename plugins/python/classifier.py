# Classifier
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

from global_logger import GlobalLogger

CAN_REGISTER_ELEMENT = True
try:
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    gi.require_version("GLib", "2.0")
    from gi.repository import Gst, GObject  # noqa: E402
    from classifier_base import ClassifierBase
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'classifier_pylm' element will not be available. Error: {e}"
    )


class Classifier(ClassifierBase):
    """
    GStreamer element for a general object detector where the user sets the model-name property.
    """

    __gstmetadata__ = (
        "Classifier",
        "Transform",
        "Classifier",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    def __init__(self):
        super().__init__()
        self.logger.info(
            "Classifier created without a model. Please set the 'model-name' property."
        )


if CAN_REGISTER_ELEMENT:
    GObject.type_register(Classifier)
    __gstelementfactory__ = ("pyml_classifier", Gst.Rank.NONE, Classifier)
else:
    GlobalLogger().warning(
        "The 'pyml_classifier' element will not be registered because a module is missing."
    )
