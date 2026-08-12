# MLEngineMixin
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

"""Portable, framework-agnostic ML element logic.

`MLEngineMixin` holds everything an ML element needs that does NOT depend on
the multimedia framework: the engine/model lifecycle and the backing storage
for the common tunable properties.

A framework backend builds a concrete element by combining this mixin with its
own element base class (`GstBase.BaseTransform`, `GstBase.Aggregator`, ...) and
declaring the properties in whatever form the framework requires
(`@GObject.Property` for GStreamer, etc). The property getters/setters read and
write the `self._<name>` fields initialised here, so the shared logic stays
identical across backends.

This module imports no `gi`; keep it that way so the same mixin can load under a
non-GStreamer backend.
"""

from engine.engine_manager import EngineManager
from log.logger_factory import LoggerFactory


class FrameProcessingMixin:
    """The per-frame work a video element does, shared by every backend.

    The backend driver extracts the frame and handles the framework's error
    signalling; what is left is the same on all of them, so it lives here once:
    infer, turn the result into a frame and/or a metadata blob, write both back.
    An element whose task class follows that contract supplies only its
    `META_HEADER`; one that does not overrides `process_frames`.
    """

    #: Tag on the metadata blob this element appends, e.g. ``b"GST-DEPTH:"``.
    #: `None` for an element that appends none.
    META_HEADER = None

    def process_frames(self, frames, num_sources, fmt, target):
        """Infer over the extracted frame(s) and write the result to `target`.

        Raises on a hard failure; the driver maps that to its own error return.
        """
        from backend import frameio

        result = self.forward(frames)
        if result is None:
            raise RuntimeError(f"{type(self).__name__}: inference returned None")
        # Inference sees the whole batch, but one buffer carries one frame back:
        # the overlay and the blob describe the primary source.
        frame = frames[0] if num_sources > 1 else frames
        if num_sources > 1 and isinstance(result, list):
            if not result:
                return
            result = result[0]
        output, blob = self.decode(frame, result, fmt)
        frameio.write_result(target, output, blob, self.META_HEADER)


class PayloadProcessingMixin:
    """The per-buffer work a non-video element does, shared by every backend.

    These families read one media type and write another (text in / text out,
    audio in / text out), so a buffer is opaque bytes rather than a frame. The
    backend driver maps the input buffer and turns whatever comes back into
    output buffers; the middle, bytes to bytes, is the same everywhere and is
    what the element supplies.
    """

    def process_payload(self, payload: bytes) -> list[bytes]:
        """Turn one input buffer's bytes into the payloads to send onward.

        Zero or more, because these families are not all 1:1: an element that
        accumulates across buffers returns an empty list until it has something
        to say, and one that chunks its output returns several at once.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement process_payload"
        )

    def payload_duration_ns(self, payload_size):
        """How long a payload of this many bytes plays for.

        `None`, the usual answer, means the output covers the same stretch of
        the stream as the input did, so it keeps the input's timing. An element
        that generates media of its own length (speech from a text buffer) works
        it out from the byte count instead.
        """
        return None


class MLEngineMixin:
    """Engine/model lifecycle and the start hook, shared by every ML element."""

    def on_start(self):
        """One-time setup before the first buffer, on any backend.

        For an element with its own background work to start (an inference
        thread, a warm-up). Framework lifecycle virtuals like `do_start` only
        exist on GStreamer, so anything a hosted element also needs goes here.
        """

    def _ensure_started(self):
        """Run `on_start` once, for a backend with no start virtual of its own."""
        if not getattr(self, "_started", False):
            self._started = True
            self.on_start()

    def _ml_init(self):
        """Initialise shared ML state. Call this from the element's __init__."""
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        self.mgr = EngineManager(self.logger)
        self.kwargs = {}
        self._batch_size = 1
        self._frame_stride = 1
        self._model_name = None
        self._device_queue_id = 0
        self._system_prompt = None
        self._prompt = None
        self._compile = False

    @property
    def engine(self):
        return self.mgr.engine

    # Engine / model lifecycle. None of this touches the framework, so every
    # backend reuses it verbatim.
    def initialize_engine(self):
        if not self.engine and self.mgr.engine_name:
            self.mgr.initialize_engine()
            self.engine.batch_size = self._batch_size
            self.engine.frame_stride = self._frame_stride
            if self._device_queue_id:
                self.engine.device_queue_id = self._device_queue_id
        if not self.engine:
            self.logger.error(f"Unsupported ML engine: {self.mgr.engine_name}")

    def do_load_model(self):
        self.initialize_engine()
        if self.engine is None:
            self.logger.error(
                f"Cannot load model {self._model_name}: engine not initialized"
            )
            return
        if self._model_name is None:
            self.logger.warning("Cannot load model as model name is not set")
            return
        self.mgr.do_load_model(self._model_name, **self.kwargs)

    def get_model(self):
        """Gets the model from the engine."""
        self.initialize_engine()
        if self.engine is None:
            self.logger.error(
                f"Cannot get model {self._model_name}: engine not initialized"
            )
            return None
        if self.engine:
            return self.engine.get_model()
        return None

    def set_model(self, model):
        """Sets the model in the engine."""
        self.initialize_engine()
        if self.engine is None:
            self.logger.error("Cannot load model: engine not initialized")
            return False
        self.engine.model = model
        self.logger.info("Model set successfully in the engine.")

    def get_tokenizer(self):
        self.initialize_engine()
        if self.engine is None:
            self.logger.error("Cannot get tokenizer: engine not initialized")
            return None
        return self.mgr.get_tokenizer()
