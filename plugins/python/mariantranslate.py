# MarianTranslate
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
    from gi.repository import Gst, GObject  # noqa: E402
    from translate_base import TranslateBase
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_mariantranslate' element will not be available. Element {e}"
    )


class MarianTranslate(TranslateBase):
    __gstmetadata__ = (
        "MarianTranslate",
        "Transform",
        "Processes text using a Large Language Model",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )


if CAN_REGISTER_ELEMENT:
    GObject.type_register(MarianTranslate)
    __gstelementfactory__ = ("pyml_mariantranslate", Gst.Rank.NONE, MarianTranslate)
else:
    GlobalLogger().warning(
        "The 'pyml_mariantranslate' element will not be registered because required modules are missing."
    )
