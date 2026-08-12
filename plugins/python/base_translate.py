# BaseTranslate
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

from abc import abstractmethod
import gi

import backend
from base_aggregator import BaseAggregator

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
from gi.repository import Gst, GObject, GstBase  # noqa: E402


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

    def __init__(self):
        super().__init__()
        self.__src = "en"
        self.__target = "en"

    @GObject.Property(type=str)
    def src(self):
        "Source language code (e.g., 'de' for German)."
        return self.__src

    @src.setter
    def src(self, value):
        self.__src = value

    @GObject.Property(type=str)
    def target(self):
        "Destination language code (e.g., 'ko' for Korean)."
        return self.__target

    @target.setter
    def target(self, value):
        self.__target = value

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
