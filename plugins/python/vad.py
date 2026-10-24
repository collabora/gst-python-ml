# VAD (Voice Activity Detection)
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from log.global_logger import GlobalLogger
import backend

CAN_REGISTER_ELEMENT = True
try:
    import struct

    from backend import GObject, InPlaceAudioTransform, frameio

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(f"The 'vad' element will not be available. Error {e}")

VAD_SAMPLE_RATE = 16000

# Header prefix for VAD metadata: "GST-VAD:" + 4-byte float32 speech probability
VAD_META_HEADER = b"GST-VAD:"


class VoiceActivityDetector(InPlaceAudioTransform):
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

    Typical use: chain before mp_whispertranscribe to pre-filter silence
    and reduce transcription latency.

    Example pipeline:
      pulsesrc ! audio/x-raw,format=S16LE,rate=16000,channels=1 \\
        ! mp_vad threshold=0.6 \\
        ! mp_whispertranscribe model-name=medium device=cuda \\
        ! fakesink
    """

    __gstmetadata__ = (
        "Voice Activity Detector",
        "Filter/Audio",
        "Detects voice activity using Silero VAD and attaches speech probability metadata",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    INPUT_CAPS = "audio/x-raw,format=S16LE,layout=interleaved,rate=16000,channels=1"

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
        self._vad = None
        self._chunk_size = None

    def on_start(self):
        from pysilero_vad import SileroVoiceActivityDetector

        self._vad = SileroVoiceActivityDetector()
        self._chunk_size = self._vad.chunk_samples()
        self.logger.info(
            f"Silero VAD initialized (chunk_size={self._chunk_size} samples "
            f"at {VAD_SAMPLE_RATE} Hz)"
        )

    def process_in_place(self, samples, buf):
        import numpy as np

        audio = samples.view(np.int16)

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
            return

        avg_score = float(np.mean(speech_scores))
        is_speech = avg_score >= self.threshold

        self.logger.debug(f"VAD score={avg_score:.3f} speech={is_speech}")

        if self.gate and not is_speech:
            samples[:] = 0

        frameio.append_blob(buf, VAD_META_HEADER, struct.pack("f", avg_score))


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        "mp_vad", VoiceActivityDetector
    )
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'mp_vad' element will not be registered because required modules are missing."
    )
