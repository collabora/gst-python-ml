# TranslateBase
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

from transformers import MarianMTModel, MarianTokenizer
from aggregator_base import AggregatorBase

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
from gi.repository import Gst, GObject, GstBase  # noqa: E402

# Define input and output caps for text/x-raw format
ICAPS = Gst.Caps(Gst.Structure("text/x-raw", format="utf8"))
OCAPS = Gst.Caps(Gst.Structure("text/x-raw", format="utf8"))


class TranslateBase(AggregatorBase):
    __gstmetadata__ = (
        "TranslateBase",
        "Aggregator",
        "Text-to-Text translation element",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    __gsttemplates__ = (
        Gst.PadTemplate.new_with_gtype(
            "sink",
            Gst.PadDirection.SINK,
            Gst.PadPresence.REQUEST,
            ICAPS,
            GstBase.AggregatorPad.__gtype__,
        ),
        Gst.PadTemplate.new_with_gtype(
            "src",
            Gst.PadDirection.SRC,
            Gst.PadPresence.ALWAYS,
            OCAPS,
            GstBase.AggregatorPad.__gtype__,
        ),
    )

    src = GObject.Property(
        type=str,
        default="en",
        nick="Source Language",
        blurb="Source language code (e.g., 'de' for German).",
    )

    target = GObject.Property(
        type=str,
        default="en",
        nick="Destination Language",
        blurb="Destination language code (e.g., 'ko' for Korean).",
    )

    def __init__(self):
        super().__init__()
        self.tokenizer = None

    def do_load_model(self):
        """
        Loads the MarianMT model based on the source
        and destination languages.
        """
        model_name = f"Helsinki-NLP/opus-mt-{self.src}-{self.target}"
        try:
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.set_model(MarianMTModel.from_pretrained(model_name))
            self.logger.info(
                f"Loaded translation model for {self.src} to {self.target}"
            )
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")

    def do_translate_text(self, text):
        """
        Translates the input text using the MarianMT model.
        """
        if self.get_model() is None or self.tokenizer is None:
            self.do_load_model()

        if self.get_model() and self.tokenizer:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True)
            translated = self.get_model().generate(**inputs)
            return self.tokenizer.decode(translated[0], skip_special_tokens=True)
        else:
            self.logger.error("Model or tokenizer is not available.")
            return ""

    def do_process(self, buf):
        """
        Processes text data from the input buffers,
        translates it, and pushes it downstream.
        """
        try:
            success, map_info = buf.map(Gst.MapFlags.READ)
            if not success:
                self.logger.error("Failed to map input buffer")
                return Gst.FlowReturn.ERROR

            byte_data = bytes(map_info.data)
            buf.unmap(map_info)

            if not byte_data:
                return Gst.FlowReturn.OK

            try:
                text_data = byte_data.decode("utf-8", errors="replace")
            except Exception as e:
                self.logger.error(f"Error decoding text data: {e}")
                return Gst.FlowReturn.ERROR

            self.logger.info(f"Translating text: {text_data}")

            # Translate the text using the MarianMT model
            translated_text = self.do_translate_text(text_data)

            if translated_text:
                self.logger.info(f"Translated text: {translated_text}")
                # Convert the translated text to a GstBuffer and push it downstream
                outbuf = self.convert_text_to_buf(translated_text, buf)
                self.finish_buffer(outbuf)

            return Gst.FlowReturn.OK

        except Exception as e:
            self.logger.error(f"Error processing text buffer: {e}")
            return Gst.FlowReturn.ERROR

    def convert_text_to_buf(self, translated_text, inbuf):
        """
        Converts translated text to a GstBuffer.
        """
        try:
            # Encode the translated text as UTF-8
            text_bytes = translated_text.encode("utf-8")
            encoded_size = len(text_bytes)

            # Create a new buffer for output and write the translated text
            outbuf = Gst.Buffer.new_allocate(None, encoded_size, None)
            outbuf.fill(0, text_bytes)

            # Set PTS and duration from the input buffer
            outbuf.pts = inbuf.pts
            outbuf.duration = inbuf.duration

            return outbuf
        except Exception as e:
            self.logger.error(f"Error converting text to buffer: {e}")
            return None
