# BaseTranslate
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from abc import abstractmethod

import backend
from base_aggregator import BaseAggregator

if backend.BACKEND == "gst":
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    from gi.repository import Gst, GstBase  # noqa: E402


class BaseTranslate(BaseAggregator):
    __gstmetadata__ = (
        "BaseTranslate",
        "Aggregator",
        "Text-to-Text translation element",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    # the caps each pad negotiates, stated once for both backends
    INPUT_CAPS = "text/x-raw,format=utf8"
    OUTPUT_CAPS = "text/x-raw,format=utf8"

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

    @abstractmethod
    def do_translate_text(self, text):
        pass

    def process_payload(self, payload: bytes) -> list[bytes]:
        """Decodes the input text, translates it, and returns the result."""
        if not payload:
            return []

        text_data = payload.decode("utf-8", errors="replace")
        self.logger.info(f"Translating text: {text_data}")

        translated_text = self.do_translate_text(text_data)
        if not translated_text:
            return []

        self.logger.info(f"Translated text: {translated_text}")
        return [translated_text.encode("utf-8")]
