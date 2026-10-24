# CaptionPhi
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

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
        f"The 'mp_caption_phi' element will not be available. Error {e}"
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
        self.mgr.engine_name = "mp_caption_phi_engine"
        EngineFactory.register(self.engine_name, CaptionPhiEngine)


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        "mp_caption_phi", CaptionPhi, "mp_caption_phi"
    )
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'mp_caption_phi' element will not be registered because required modules are missing."
    )
