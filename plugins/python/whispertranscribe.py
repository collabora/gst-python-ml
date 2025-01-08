# WhisperTranscribe
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

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("GObject", "2.0")
from gi.repository import Gst, GObject  # noqa: E402

CAN_REGISTER_ELEMENT = True
try:
    from gst_transcribe import GstTranscribe
    from faster_whisper import WhisperModel
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    Gst.warning(f"The 'whispertranscribe' element will not be available. Error: {e}")


class WhisperTranscribe(GstTranscribe):
    __gstmetadata__ = (
        "WhisperTranscribe",
        "Text Output",
        "Python element that transcribes audio with Whisper",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    def __init__(self):
        super(WhisperTranscribe, self).__init__()
        self.model_name = "medium"

    def do_load_model(self):
        compute_type = "float16" if self.device.startswith("cuda") else "int8"
        Gst.info(
            f"Loading Whisper model on device: {self.device} with compute_type: {compute_type}"
        )
        # Set the model and ensure it is not None
        self.set_model(
            WhisperModel(self.model_name, device=self.device, compute_type=compute_type)
        )
        if self.get_model() is None:
            Gst.error("Failed to load Whisper model.")
        else:
            Gst.info(f"Whisper model loaded successfully on {self.device}")
        self.old_device = self.device

    def do_transcribe(self, audio_data, task):
        result, _ = self.get_model().transcribe(
            audio_data,
            language=self.language,
            task=task,
            initial_prompt=self.initial_prompt,
        )
        return result


if CAN_REGISTER_ELEMENT:
    GObject.type_register(WhisperTranscribe)
    __gstelementfactory__ = ("whispertranscribe_pyml", Gst.Rank.NONE, WhisperTranscribe)
else:
    Gst.warning(
        "The 'whispertranscribe_pyml' element will not be registered because gst_transcribe module is missing."
    )
