# CaptionPhi
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
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

from log.global_logger import GlobalLogger
import backend

from base_caption import BaseCaption

CAN_REGISTER_ELEMENT = True
try:
    from engine.caption_phi_engine import CaptionPhiEngine
    from engine.engine_factory import EngineFactory

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_caption_phi' element will not be available. Error {e}"
    )


class CaptionPhi(BaseCaption):
    """
    GStreamer element for captioning video frames using Phi Vision.
    """

    __gstmetadata__ = (
        "CaptionPhi",
        "Transform",
        "Captions video clips using Phi Vision model",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    def __init__(self):
        super().__init__()
        # set engine_name directly on mgr, as engine_name property is read only
        self.mgr.engine_name = "pyml_caption_phi_engine"
        EngineFactory.register(self.engine_name, CaptionPhiEngine)


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        "pyml_caption_phi", CaptionPhi, "pyml_caption_phi"
    )
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'pyml_caption_phi' element will not be registered because required modules are missing."
    )
