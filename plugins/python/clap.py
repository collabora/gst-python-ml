# CLAP (Contrastive Language-Audio Pretraining)
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
    import json

    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    from gi.repository import Gst, GstBase
    from backend import GObject

    from log.logger_factory import LoggerFactory
    from engine.clap_engine import ClapEngine
    from engine.engine_factory import EngineFactory

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(f"The 'clap' element will not be available. Error {e}")

# Header prefix for CLAP classification metadata
CLAP_META_HEADER = b"GST-CLAP:"


class ClapTransform(GstBase.BaseTransform):
    """
    GStreamer element for audio event classification using CLAP.

    Accepts 48 kHz mono F32LE audio. For each buffer, encodes audio with CLAP
    and compares against a configurable set of text labels. Results are appended
    as a GST-CLAP: JSON memory chunk on the buffer.

    Downstream elements can read classifications:
      for i in range(buf.n_memory()):
          data = bytes(buf.peek_memory(i).map(Gst.MapFlags.READ).data)
          if data.startswith(b"GST-CLAP:"):
              results = json.loads(data[9:])

    Example pipeline:
      filesrc location=audio.wav ! decodebin ! audioconvert ! audioresample \\
        ! audio/x-raw,format=F32LE,rate=48000,channels=1 \\
        ! pyml_clap labels="gunshot,siren,baby crying,music,speech" \\
        ! fakesink
    """

    __gstmetadata__ = (
        "CLAP Audio Classifier",
        "Filter/Audio",
        "Audio event classification via CLAP contrastive language-audio pretraining",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    AUDIO_CAPS = Gst.Caps.from_string(
        "audio/x-raw,format=F32LE,layout=interleaved,rate=48000,channels=1"
    )

    sink_template = Gst.PadTemplate.new(
        "sink", Gst.PadDirection.SINK, Gst.PadPresence.ALWAYS, AUDIO_CAPS
    )
    src_template = Gst.PadTemplate.new(
        "src", Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS, AUDIO_CAPS
    )
    __gsttemplates__ = (sink_template, src_template)

    model_name = GObject.Property(
        type=str,
        default="laion/larger_clap_music_and_speech",
        nick="Model Name",
        blurb="HuggingFace model ID for CLAP",
        flags=GObject.ParamFlags.READWRITE,
    )

    threshold = GObject.Property(
        type=float,
        default=0.3,
        minimum=0.0,
        maximum=1.0,
        nick="Confidence Threshold",
        blurb="Minimum similarity score to include a label in results",
        flags=GObject.ParamFlags.READWRITE,
    )

    draw_label = GObject.Property(
        type=bool,
        default=True,
        nick="Draw Label",
        blurb="Include human-readable label string in metadata output",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self.set_in_place(True)
        self._engine = None
        self._labels_str = "gunshot,siren,baby crying,music,speech"
        self._labels = [label.strip() for label in self._labels_str.split(",")]
        EngineFactory.register("pyml_clap_engine", ClapEngine)

    @GObject.Property(type=str)
    def labels(self):
        """Comma-separated text queries for audio classification."""
        return self._labels_str

    @labels.setter
    def labels(self, value):
        self._labels_str = value
        self._labels = [label.strip() for label in value.split(",") if label.strip()]
        if self._engine is not None and self._engine.processor is not None:
            self._engine._precompute_text_embeddings(self._labels)

    def do_start(self):
        try:
            self._engine = EngineFactory.create("pyml_clap_engine")
            self._engine.do_load_model(self.model_name, labels=self._labels)
            self.logger.info(f"CLAP element started with {len(self._labels)} labels")
        except Exception as e:
            self.logger.error(f"Failed to start CLAP element: {e}")
            return False
        return True

    def do_stop(self):
        self._engine = None
        return True

    def do_transform_ip(self, buf):
        import numpy as np

        if self._engine is None:
            return Gst.FlowReturn.OK

        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            self.logger.error("Failed to map audio buffer for reading")
            return Gst.FlowReturn.ERROR

        try:
            audio = np.frombuffer(bytes(map_info.data), dtype=np.float32)
        finally:
            buf.unmap(map_info)

        if len(audio) == 0:
            return Gst.FlowReturn.OK

        results = self._engine.do_forward(audio)
        if results is None:
            return Gst.FlowReturn.OK

        # Filter by threshold
        filtered = [
            {"label": label, "score": round(score, 4)}
            for label, score in results
            if score >= self.threshold
        ]

        if not filtered and self.draw_label:
            # Include the top result even if below threshold when draw_label is set
            if results:
                top_label, top_score = results[0]
                filtered = [{"label": top_label, "score": round(top_score, 4)}]

        # Append classification metadata as a JSON memory chunk.
        # Use new_allocate+fill: PyGI hides the maxsize arg in new_wrapped
        # (it derives it from data length), so passing it explicitly shifts
        # all subsequent args and causes a GI assertion crash.
        meta_bytes = CLAP_META_HEADER + json.dumps(filtered).encode("utf-8")
        tmp = Gst.Buffer.new_allocate(None, len(meta_bytes), None)
        tmp.fill(0, meta_bytes)
        buf.append_memory(tmp.get_memory(0))

        if filtered:
            top = filtered[0]
            self.logger.debug(f"CLAP top match: {top['label']} ({top['score']:.3f})")

        return Gst.FlowReturn.OK


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element("pyml_clap", ClapTransform)
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'pyml_clap' element will not be registered because required modules are missing."
    )
