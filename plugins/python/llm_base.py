# LlmBase
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

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("GLib", "2.0")

from gi.repository import Gst  # noqa: E402

from aggregator_base import AggregatorBase


class LlmBase(AggregatorBase):
    """
    GStreamer base element that performs language model inference
    with a PyTorch model.
    """

    __gsttemplates__ = (
        Gst.PadTemplate.new(
            "src",
            Gst.PadDirection.SRC,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.from_string("text/x-raw,format=utf8"),
        ),
        Gst.PadTemplate.new(
            "sink",
            Gst.PadDirection.SINK,
            Gst.PadPresence.REQUEST,
            Gst.Caps.from_string("text/x-raw,format=utf8"),
        ),
    )

    def __init__(self):
        super().__init__()
        self.engine = None  # Explicitly initialize self.engine

    def do_process(self, buf):
        """
        Processes the input buffer with the language model
        and pushes the result downstream.
        """
        try:
            # Map buffer to read input text
            success, map_info = buf.map(Gst.MapFlags.READ)
            if not success:
                self.logger.error("Failed to map buffer")
                return Gst.FlowReturn.ERROR

            # Convert memoryview to bytes and decode to string
            input_text = bytes(map_info.data).decode("utf-8")
            self.logger.info(f"Received text for LLM processing: {input_text}")

            # Ensure engine is initialized
            if not self.engine_helper.engine:
                self.logger.info("Engine not initialized, initializing now")
                self.engine_helper.initialize_engine(self.engine_name)
                self.engine_helper.load_model(self.model_name)
                self.engine = self.engine_helper.engine  # Set self.engine

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
                if not self.engine_helper.load_model(self.model_name):
                    self.logger.error("Model reload failed")
                    buf.unmap(map_info)
                    return Gst.FlowReturn.ERROR
                self.engine = self.engine_helper.engine  # Update self.engine
                tokenizer = self.get_tokenizer()
                model = self.get_model()
                if not tokenizer or not model:
                    self.logger.error("Model reload failed again")
                    buf.unmap(map_info)
                    return Gst.FlowReturn.ERROR

            # Generate text using the engine
            generated_text = self.engine.generate(input_text)
            self.logger.info(f"Generated text: {generated_text}")

            buf.unmap(map_info)

            # Push the generated text downstream
            self.push_generated_text(generated_text)

            return Gst.FlowReturn.OK

        except Exception as e:
            self.logger.error(f"Error in LLM processing: {e}")
            return Gst.FlowReturn.ERROR

    def push_generated_text(self, generated_text):
        """
        Push the generated text downstream.
        """
        try:
            generated_bytes = generated_text.encode("utf-8")
            outbuf = Gst.Buffer.new_allocate(None, len(generated_bytes), None)
            success, map_info_out = outbuf.map(Gst.MapFlags.WRITE)
            if not success:
                self.logger.error("Failed to map output buffer for writing")
                return Gst.FlowReturn.ERROR

            map_info_out.data[: len(generated_bytes)] = generated_bytes
            outbuf.unmap(map_info_out)

            # Push the buffer downstream
            self.srcpad.push(outbuf)
            self.logger.info("Pushed generated text downstream")

        except Exception as e:
            self.logger.error(f"Error pushing generated text: {e}")
