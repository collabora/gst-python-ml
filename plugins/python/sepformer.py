# Sepformer (assuming this is sepformer.py based on traceback)
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from log.global_logger import GlobalLogger
import backend

CAN_REGISTER_ELEMENT = True
try:
    from backend import GObject
    from base_separate import BaseSeparate

    from engine.sepformer_engine import SepformerEngine
    from engine.engine_factory import EngineFactory

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'mp_sepformer' element will not be available. Error: {e}"
    )

if backend.BACKEND == "gst":
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    from gi.repository import Gst, GstBase  # noqa: E402


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
        self.model_name = "mp_sepformer_model"
        self.mgr.engine_name = "mp_sepformer_engine"
        EngineFactory.register(self.mgr.engine_name, SepformerEngine)

    @GObject.Property(type=str)
    def engine_name(self):
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        raise ValueError("engine_name cannot be set")

    def do_separate(self, audio_data):
        import torch

        engine = self.engine
        if engine.model is None:
            engine.do_load_model(self.model_name)

        audio_torch = torch.from_numpy(audio_data).float().to(engine.device)
        mixture = audio_torch.unsqueeze(0)  # (1, length) for SpeechBrain

        sources = engine.separate_sources(
            mixture,
            segment=(
                15.0 if not self.streaming else 1.0
            ),  # Increased for better quality
            overlap=0.1,
        )  # (batch, sources, length)
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
    __gstelementfactory__ = backend.register_gst_element("mp_sepformer", Sepformer)
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning("mp_sepformer not registered")
