# BaseLlm
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


import backend
from backend import GObject
from base_aggregator import BaseAggregator

if backend.BACKEND == "gst":
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst  # noqa: E402


class BaseLlm(BaseAggregator):
    """
    Base element that performs language model inference with a PyTorch model.
    """

    PUSH_FROM_SRC_PAD = True

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

    # the caps each pad negotiates, stated once for both backends
    INPUT_CAPS = "text/x-raw,format=utf8"
    OUTPUT_CAPS = "text/x-raw,format=utf8"

    # Building a Gst object needs Gst.init, which only the gst backend calls.
    if backend.BACKEND == "gst":
        __gsttemplates__ = (
            Gst.PadTemplate.new(
                "src",
                Gst.PadDirection.SRC,
                Gst.PadPresence.ALWAYS,
                Gst.Caps.from_string(OUTPUT_CAPS),
            ),
            Gst.PadTemplate.new(
                "sink",
                Gst.PadDirection.SINK,
                Gst.PadPresence.REQUEST,
                Gst.Caps.from_string(INPUT_CAPS),
            ),
        )

    def process_payload(self, payload: bytes) -> list[bytes]:
        """Generates a reply to the input text with the language model."""
        input_text = payload.decode("utf-8")
        self.logger.info(f"Received text for LLM processing: {input_text}")

        # Ensure engine is initialized
        if not self.engine:
            self.logger.info("Engine not initialized, initializing now")
            self.mgr.initialize_engine()
            self.mgr.do_load_model(self.model_name)

        # Retry model loading if tokenizer or model is missing
        tokenizer = self.get_tokenizer()
        model = self.get_model()
        self.logger.info(f"Tokenizer: {tokenizer}")
        self.logger.info(f"Model: {model}")
        if not tokenizer or not model:
            self.logger.error(
                f"Tokenizer initialized: {tokenizer is not None}, Model initialized: {model is not None}"
            )
            self.logger.warning("Attempting to reload model")
            if not self.mgr.do_load_model(self.model_name):
                self.logger.error("Model reload failed")
                return []
            tokenizer = self.get_tokenizer()
            model = self.get_model()
            if not tokenizer or not model:
                self.logger.error("Model reload failed again")
                return []

        generated_text = self.engine.do_generate(
            input_text, system_prompt=self.system_prompt
        )
        self.logger.info(f"Generated text: {generated_text}")

        return [generated_text.encode("utf-8")]
