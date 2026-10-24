# GstStableDiffusion
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

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
