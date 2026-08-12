# BaseCaption
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


import backend
from backend import GObject, analytics
from video_transform import VideoTransform

# The text pad is a GStreamer request pad: a hosted element on g2g has one
# source pad and stages its caption as metadata instead.
if backend.BACKEND == "gst":
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    from gi.repository import Gst, GstBase  # noqa: E402

    TEXT_CAPS = Gst.Caps.from_string("text/x-raw, format=utf8")

#: How long a caption stays on screen. Long enough that the subtitle is still up
#: when the next frame's caption arrives.
CAPTION_DURATION_SECONDS = 60


class BaseCaption(VideoTransform):
    """
    Base element for captioning video frames.
    """

    __gstmetadata__ = (
        "BaseCaption",
        "Transform",
        "Captions video clips",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    if backend.BACKEND == "gst":
        __gsttemplates__ = (
            Gst.PadTemplate.new(
                "text_src", Gst.PadDirection.SRC, Gst.PadPresence.REQUEST, TEXT_CAPS
            ),
        )

    def __init__(self):
        super().__init__()
        self._prompt = "What is shown in this image?"
        self.text_src_pad = None

    @GObject.Property(type=str)
    def system_prompt(self):
        "Custom system prompt text"
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value):
        self._system_prompt = value

    @GObject.Property(type=str)
    def prompt(self):
        "Custom prompt text"
        return self._prompt

    @prompt.setter
    def prompt(self, value):
        self._prompt = value

    # make read only
    @GObject.Property(type=str)
    def engine_name(self):
        "Machine Learning Engine to use : pytorch, tflite, tensorflow, onnx or openvino, or custom engine name"
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        raise ValueError(
            "The 'engine_name' property cannot be set in this derived class."
        )

    def forward(self, frames):
        return self.engine.do_forward(frames) if self.engine else None

    def process_frames(self, frames, num_sources, fmt, target):
        """Caption each source, staging one classification per caption.

        Captions carry no pixels and no blob, so this replaces the shared
        infer-decode-write body rather than filling in `decode`.
        """
        result = self.forward(frames)
        if result is None:
            raise RuntimeError(f"{type(self).__name__}: captioning returned None")

        captions = result if isinstance(result, list) else [result] * num_sources
        if len(captions) != num_sources:
            raise RuntimeError(f"expected {num_sources} captions, got {len(captions)}")

        meta = analytics.add_relation_meta(target)
        if meta is None:
            self.logger.error("Failed to add analytics metadata to buffer")
            return

        for index, caption in enumerate(captions):
            if not caption:
                self.logger.warning(f"stream {index}: no caption generated")
                continue
            label = caption if num_sources == 1 else f"stream_{index}_{caption}"
            if analytics.add_classification(meta, index, label) is None:
                self.logger.error(f"stream {index}: failed to add the caption")
            else:
                self.logger.info(f"stream {index}: added caption {caption}")

        self.push_captions(captions, target)

    def push_captions(self, captions, buf):
        """Send each caption out the text pad, one buffer per source.

        The caption is staged as metadata either way, so the pad is an extra:
        nothing to do unless something asked for it, which is also what makes
        this a no-op on a backend that has no request pads.
        """
        if self.text_src_pad is None:
            return

        if buf.pts == Gst.CLOCK_TIME_NONE:
            buf.pts = Gst.util_uint64_scale(
                Gst.util_get_timestamp(),
                1,  # framerate_denom
                30 * Gst.SECOND,  # framerate_num
            )
        if buf.duration == Gst.CLOCK_TIME_NONE:
            buf.duration = Gst.SECOND // 30  # framerate_num

        share = buf.duration // len(captions)
        for index, caption in enumerate(captions):
            if caption:
                self.push_text_buffer(caption, buf.pts + index * share, buf.dts)

    def push_text_buffer(self, text, pts, dts):
        """Push one caption to the `text_src` pad, timed with its video frame."""
        text_buffer = Gst.Buffer.new_wrapped(text.encode("utf-8"))
        text_buffer.pts = pts
        text_buffer.dts = dts
        text_buffer.duration = CAPTION_DURATION_SECONDS * Gst.SECOND

        ret = self.text_src_pad.push(text_buffer)
        if ret != Gst.FlowReturn.OK:
            self.logger.warning(f"Failed to push text buffer: {ret}")

    def do_request_new_pad(self, template, name, caps):
        if self.text_src_pad:
            self.logger.error("Element already has a text_src")
            return None
        if name != template.name_template:
            self.logger.error("Invalid pad name")
            return None

        self.text_src_pad = Gst.Pad.new_from_template(template, name)
        self.add_pad(self.text_src_pad)
        self.text_src_pad.set_active(True)

        return self.text_src_pad

    def do_release_pad(self, pad):
        self.remove_pad(pad)
        pad.set_active(False)
        self.text_src_pad = None

    def do_sink_event(self, event):
        if self.text_src_pad:
            text_event = (
                Gst.Event.new_caps(TEXT_CAPS)
                if event.type == Gst.EventType.CAPS
                else event
            )
            self.text_src_pad.push_event(text_event)
        return GstBase.BaseTransform.do_sink_event(self, event)
