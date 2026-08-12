# Sepformer (assuming this is sepformer.py based on traceback)
# Copyright (C) 2024-2026 Collabora Ltd.
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

from log.global_logger import GlobalLogger
import backend

CAN_REGISTER_ELEMENT = True
try:
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    gi.require_version("GObject", "2.0")
    from gi.repository import Gst, GstBase  # noqa: E402
    from backend import GObject
    from base_separate import BaseSeparate

    from engine.sepformer_engine import SepformerEngine
    from engine.engine_factory import EngineFactory

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_sepformer' element will not be available. Error: {e}"
    )


class Sepformer(BaseSeparate):
    __gstmetadata__ = (
        "Sepformer",
        "Audio Output",
        "Python element that separates speakers with RE-SepFormer in noisy environments",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    SAMPLE_RATE = 8000

    INPUT_CAPS = "audio/x-raw,format=S16LE,layout=interleaved,rate=8000,channels=1"
    OUTPUT_CAPS = "audio/x-raw,format=S16LE,layout=interleaved,rate=8000,channels=1"

    # Building a Gst object needs Gst.init, which only the gst backend calls.
    if backend.BACKEND == "gst":
        __gsttemplates__ = (
            Gst.PadTemplate.new_with_gtype(
                "sink",
                Gst.PadDirection.SINK,
                Gst.PadPresence.REQUEST,
                Gst.Caps.from_string(INPUT_CAPS),
                GstBase.AggregatorPad.__gtype__,
            ),
            Gst.PadTemplate.new_with_gtype(
                "src",
                Gst.PadDirection.SRC,
                Gst.PadPresence.ALWAYS,
                Gst.Caps.from_string(OUTPUT_CAPS),
                GstBase.AggregatorPad.__gtype__,
            ),
        )

    def __init__(self):
        super().__init__()
        self.model_name = "pyml_sepformer_model"
        self.mgr.engine_name = "pyml_sepformer_engine"
        EngineFactory.register(self.mgr.engine_name, SepformerEngine)

    @GObject.Property(type=str)
    def engine_name(self):
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        raise ValueError("engine_name cannot be set")

    def do_separate(self, audio_data):
        import numpy as np
        import torch

        engine = self.engine
        if engine.model is None:
            engine.do_load_model(self.model_name)

        audio_torch = torch.from_numpy(audio_data).float().to(engine.device)
        mixture = audio_torch.unsqueeze(0)  # (1, length) for SpeechBrain

        try:
            sources = engine.separate_sources(
                mixture,
                segment=(
                    15.0 if not self.streaming else 1.0
                ),  # Increased for better quality
                overlap=0.1,
            )  # (batch, sources, length)
        except Exception as e:
            self.logger.error(f"Separation failed: {e}")
            return np.zeros(len(audio_data), dtype=np.float32)

        energies = torch.mean(sources**2, dim=[1, 2])
        if energies.sum() == 0:
            idx = 0  # Default to first source if all energies zero
        else:
            idx = torch.argmax(energies)
        selected = sources[0, idx, :]  # batch=0

        # Normalize to unit amplitude for better volume
        max_amp = selected.abs().max()
        if max_amp > 0:
            selected = selected / max_amp

        return selected.cpu().numpy()


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element("pyml_sepformer", Sepformer)
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning("pyml_sepformer not registered")
