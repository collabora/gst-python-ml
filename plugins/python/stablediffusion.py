# GstStableDiffusion
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

from log.global_logger import GlobalLogger
import backend

CAN_REGISTER_ELEMENT = True
try:
    from base_aggregator import BaseAggregator
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'mp_stablediffusion' element will not be available. Error: {e}"
    )

if backend.BACKEND == "gst":
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    from gi.repository import Gst, GstBase  # noqa: E402


class StableDiffusion(BaseAggregator):
    __gstmetadata__ = (
        "StableDiffusion",
        "Aggregator",
        "Generates images from text using Stable Diffusion",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    PUSH_FROM_SRC_PAD = True

    # the caps each pad negotiates, stated once for both backends
    INPUT_CAPS = "text/x-raw,format=utf8"
    OUTPUT_CAPS = "video/x-raw,width=512,height=512,format=RGBA,framerate=0/1"

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
            Gst.PadTemplate.new(
                "src",
                Gst.PadDirection.SRC,
                Gst.PadPresence.ALWAYS,
                Gst.Caps.from_string(OUTPUT_CAPS),
            ),
        )

    def do_load_model(self):
        import torch
        from diffusers import StableDiffusionPipeline

        self.logger.info(f"Initializing Stable Diffusion model on {self.device}")
        # fp32 weights do not fit a 6 GB card
        dtype = torch.float16 if str(self.device).startswith("cuda") else torch.float32
        self.set_model(
            StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4", torch_dtype=dtype
            )
        )
        self.get_model().to(self.device)

    def process_payload(self, payload: bytes) -> list[bytes]:
        prompt = payload.decode("utf-8", errors="replace")
        if not prompt:
            return []
        self.logger.info(f"Generating image for prompt: {prompt}")
        image = self.get_model()(prompt).images[0]
        return [image.convert("RGBA").tobytes()]


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        "mp_stablediffusion", StableDiffusion
    )
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'mp_stablediffusion' element will not be registered because required modules are missing."
    )
