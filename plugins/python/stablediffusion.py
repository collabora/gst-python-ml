# GstStableDiffusion
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

from global_logger import GlobalLogger

CAN_REGISTER_ELEMENT = True
try:
    import asyncio
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    gi.require_version("GObject", "2.0")
    from gi.repository import Gst, GObject, GstBase  # noqa: E402
    from aggregator_base import AggregatorBase
    import numpy as np
    from diffusers import StableDiffusionPipeline
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_stablediffusion' element will not be available. Error: {e}"
    )

# Set output caps to image format (e.g., PNG)
ICAPS = Gst.Caps(Gst.Structure("text/plain", format="utf8"))


class StableDiffusion(AggregatorBase):
    __gstmetadata__ = (
        "StableDiffusion",
        "Aggregator",
        "Generates images from text using Stable Diffusion",
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
        Gst.PadTemplate.new(
            "src",
            Gst.PadDirection.SRC,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.from_string(
                "video/x-raw, width=512, height=512, format=RGBA, framerate=0/1"
            ),
        ),
    )

    def do_load_model(self):
        """
        Initialize the Stable Diffusion model
        """
        self.logger.info(f"Initializing Stable Diffusion model on {self.device}")
        self.set_model(
            StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        )
        self.get_model().to(self.device)

    def do_process(self, buf):
        try:
            success, map_info = buf.map(Gst.MapFlags.READ)
            if not success:
                self.logger.error("Failed to map input buffer")
                return Gst.FlowReturn.ERROR

            byte_data = bytes(map_info.data)
            if not byte_data:
                buf.unmap(map_info)
                return Gst.FlowReturn.OK

            try:
                byte_data = byte_data.decode("utf-8", errors="replace")
            except Exception as e:
                self.logger.error(f"Error decoding text data: {e}")
                buf.unmap(map_info)
                return Gst.FlowReturn.ERROR

            self.logger.info(f"Text to Image: received text: {byte_data}")

            # Generate the image asynchronously
            self.convert_text_to_image_async(byte_data)

            buf.unmap(map_info)

        except Exception as e:
            self.logger.error(f"Error processing text buffer: {e}")
            return Gst.FlowReturn.ERROR

    async def process_text(self, text):
        try:
            if self.get_model() is None:
                self.do_load_model()

            image_data = self.generate_image(text)

            # Push the raw image buffer downstream
            self.push_image_to_pipeline(image_data)

        except Exception as e:
            self.logger.error(f"Error processing text to image: {e}")

    def convert_text_to_image_async(self, text):
        asyncio.run(self.process_text(text))

    def generate_image(self, prompt):
        """
        Generate image from text prompt using Stable Diffusion.
        """
        self.logger.info(f"Generating image for prompt: {prompt}")
        image = self.get_model()(prompt).images[0]  # Generate image for the prompt

        # Convert the PIL image to raw RGBA format
        image = image.convert("RGBA")  # Convert to RGBA format
        image_data = np.array(image)  # Convert to a NumPy array (RGBA format)
        return image_data

    def push_image_to_pipeline(self, image_data):
        try:
            # Create a new GStreamer buffer with the raw RGBA image data
            buffer = Gst.Buffer.new_wrapped(image_data.tobytes())

            # Set buffer PTS and duration
            buffer.pts = Gst.CLOCK_TIME_NONE
            buffer.duration = Gst.CLOCK_TIME_NONE

            # Push the buffer downstream to the next element (e.g., pngenc for encoding)
            ret = self.srcpad.push(buffer)
            if ret != Gst.FlowReturn.OK:
                raise RuntimeError(f"Error pushing image to pipeline: {ret}")

            self.logger.info("Raw image generated and pushed downstream successfully.")

        except Exception as e:
            self.logger.error(f"Error pushing image to pipeline: {e}")


if CAN_REGISTER_ELEMENT:
    GObject.type_register(StableDiffusion)
    __gstelementfactory__ = (
        "pyml_stablediffusion",
        Gst.Rank.NONE,
        StableDiffusion,
    )
else:
    GlobalLogger().warning(
        "The 'pyml_stablediffusion' element will not be registered because required modules are missing."
    )
