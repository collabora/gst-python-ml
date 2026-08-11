# VAD (Voice Activity Detection)
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
    import ctypes
    import struct

    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    from gi.repository import Gst, GstBase
    from backend import GObject

    from log.logger_factory import LoggerFactory

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(f"The 'vad' element will not be available. Error {e}")

VAD_SAMPLE_RATE = 16000

# Header prefix for VAD metadata: "GST-VAD:" + 4-byte float32 speech probability
VAD_META_HEADER = b"GST-VAD:"


class VoiceActivityDetector(GstBase.BaseTransform):
    """
    GStreamer element for standalone Voice Activity Detection (VAD).

    Accepts 16kHz mono S16LE audio. For each buffer, runs Silero VAD and:
      1. Appends speech probability as a GST-VAD: buffer memory chunk.
      2. When gate=True, zeroes out (mutes) audio classified as silence.

    Downstream elements can read the speech probability:
      for i in range(buf.n_memory()):
          data = bytes(buf.peek_memory(i).map(Gst.MapFlags.READ).data)
          if data.startswith(b"GST-VAD:"):
              prob = struct.unpack("f", data[8:12])[0]

    Typical use: chain before pyml_whispertranscribe to pre-filter silence
    and reduce transcription latency.

    Example pipeline:
      pulsesrc ! audio/x-raw,format=S16LE,rate=16000,channels=1 \\
        ! pyml_vad threshold=0.6 \\
        ! pyml_whispertranscribe model-name=medium device=cuda \\
        ! fakesink
    """

    __gstmetadata__ = (
        "Voice Activity Detector",
        "Filter/Audio",
        "Detects voice activity using Silero VAD and attaches speech probability metadata",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    AUDIO_CAPS = Gst.Caps.from_string(
        "audio/x-raw,format=S16LE,layout=interleaved,rate=16000,channels=1"
    )

    sink_template = Gst.PadTemplate.new(
        "sink", Gst.PadDirection.SINK, Gst.PadPresence.ALWAYS, AUDIO_CAPS
    )
    src_template = Gst.PadTemplate.new(
        "src", Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS, AUDIO_CAPS
    )
    __gsttemplates__ = (sink_template, src_template)

    threshold = GObject.Property(
        type=float,
        default=0.7,
        minimum=0.0,
        maximum=1.0,
        nick="Speech Threshold",
        blurb="Minimum VAD confidence to classify audio as speech (0.0-1.0)",
        flags=GObject.ParamFlags.READWRITE,
    )

    gate = GObject.Property(
        type=bool,
        default=False,
        nick="Gate Audio",
        blurb="When True, zero out (mute) audio buffers classified as silence",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self.set_in_place(True)
        self._vad = None
        self._chunk_size = None

    def do_start(self):
        try:
            from pysilero_vad import SileroVoiceActivityDetector

            self._vad = SileroVoiceActivityDetector()
            self._chunk_size = self._vad.chunk_samples()
            self.logger.info(
                f"Silero VAD initialized (chunk_size={self._chunk_size} samples "
                f"at {VAD_SAMPLE_RATE} Hz)"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize Silero VAD: {e}")
            return False
        return True

    def do_stop(self):
        self._vad = None
        self._chunk_size = None
        return True

    def do_transform_ip(self, buf):
        import numpy as np

        if self._vad is None:
            return Gst.FlowReturn.OK

        # Read audio samples
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            self.logger.error("Failed to map audio buffer for reading")
            return Gst.FlowReturn.ERROR

        try:
            audio = np.frombuffer(bytes(map_info.data), dtype=np.int16)
        finally:
            buf.unmap(map_info)

        # Compute average speech probability across VAD chunks
        speech_scores = []
        offset = 0
        while offset + self._chunk_size <= len(audio):
            chunk = audio[offset : offset + self._chunk_size]
            score = self._vad.process_chunk(chunk.tobytes())
            speech_scores.append(score)
            offset += self._chunk_size

        if not speech_scores:
            # Buffer too short for a full chunk; pass through without metadata
            return Gst.FlowReturn.OK

        avg_score = float(np.mean(speech_scores))
        is_speech = avg_score >= self.threshold

        self.logger.debug(f"VAD score={avg_score:.3f} speech={is_speech}")

        # Gate: mute audio BEFORE appending read-only metadata memory.
        # (A READONLY chunk on the buffer would prevent buf.map(WRITE) from succeeding.)
        if self.gate and not is_speech:
            success_w, map_info_w = buf.map(Gst.MapFlags.WRITE)
            if success_w:
                try:
                    dst = (ctypes.c_char * map_info_w.size).from_buffer(map_info_w.data)
                    ctypes.memset(dst, 0, map_info_w.size)
                finally:
                    buf.unmap(map_info_w)

        # Append VAD metadata: header + 4-byte float32 speech probability.
        # Use new_allocate+fill: PyGI hides the maxsize arg in new_wrapped
        # (it derives it from data length), so passing it explicitly shifts
        # all subsequent args and causes a GI assertion crash.
        vad_bytes = VAD_META_HEADER + struct.pack("f", avg_score)
        tmp = Gst.Buffer.new_allocate(None, len(vad_bytes), None)
        tmp.fill(0, vad_bytes)
        buf.append_memory(tmp.get_memory(0))

        return Gst.FlowReturn.OK


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        "pyml_vad", VoiceActivityDetector
    )
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'pyml_vad' element will not be registered because required modules are missing."
    )
