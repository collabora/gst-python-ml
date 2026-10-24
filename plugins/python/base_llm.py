# BaseLlm
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


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

        generated_text = self.engine.do_generate(
            input_text, system_prompt=self.system_prompt
        )
        self.logger.info(f"Generated text: {generated_text}")

        return [generated_text.encode("utf-8")]
