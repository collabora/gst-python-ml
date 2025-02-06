# WhisperLive
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
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    gi.require_version("GObject", "2.0")
    from gi.repository import Gst, GObject, GstBase  # noqa: E402
    from transcribe_base import TranscribeBase
    from whisperspeech.pipeline import Pipeline
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import numpy as np
    import torch
    from faster_whisper import WhisperModel
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_whisperlive' element will not be available. Error: {e}"
    )

TTS_SAMPLE_RATE = 24000
model_ref = "collabora/whisperspeech:s2a-q4-base-en+pl.model"


OCAPS = Gst.Caps(
    Gst.Structure(
        "audio/x-raw",
        format="S16LE",
        layout="interleaved",
        rate=TTS_SAMPLE_RATE,
        channels=1,
    )
)


class WhisperLive(TranscribeBase):
    __gstmetadata__ = (
        "WhisperLive",
        "Text Output",
        "Python element that transcribes audio with Whisper",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    __gsttemplates__ = Gst.PadTemplate.new_with_gtype(
        "src",
        Gst.PadDirection.SRC,
        Gst.PadPresence.ALWAYS,
        OCAPS,
        GstBase.AggregatorPad.__gtype__,
    )

    llm_model_name = GObject.Property(
        type=str,
        default="microsoft/phi-2",
        nick="LLM Model Name",
        blurb="Name of the pre-trained model to load from Hugging Face hub",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.model_name = "medium"
        self.pipeline = None
        self.llm_tokenizer = None
        self.llm_model = None

    def do_get_property(self, prop: GObject.ParamSpec):
        if prop.name == "llm-model-name":
            return self.llm_model_name
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def do_set_property(self, prop: GObject.ParamSpec, value):
        if prop.name == "llm-model-name":
            self.llm_model_name = value
            self.do_load_model()  # Load model whenever the model name is set or changed
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def do_load_model(self):
        """Initialize or update the Whisper model when the device changes."""
        compute_type = "float16" if self.device.startswith("cuda") else "int8"
        self.logger.info(
            f"Loading Whisper model on device: {self.device} with compute_type: {compute_type}"
        )
        self.set_model(
            WhisperModel(self.model_name, device=self.device, compute_type=compute_type)
        )
        self.old_device = self.device

        # load WhisperSpeech TTS model
        self.logger.info(
            f"Initializing WhisperSpeech TTS model on device: {self.device}"
        )
        try:
            self.pipeline = Pipeline(
                s2a_ref=model_ref, device=self.device, torch_compile=True
            )
            if self.pipeline is not None:
                self.logger.info(
                    f"WhisperSpeech pipeline initialized successfully: {self.get_model()}"
                )
            else:
                self.logger.error("Failed to create WhisperSpeech pipeline")
        except Exception as e:
            self.logger.error(f"Exception during model initialization: {e}")

        # load LLM
        """
        Load the tokenizer and model using the specified model path or a default model.
        """
        if not self.llm_model_name:
            self.logger.error("LLM model name is not set. Cannot load model.")
            return

        try:
            self.logger.info(f"Loading model and tokenizer for: {self.llm_model_name}")
            self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto",
            )
            self.llm_model.eval()
            self.logger.info(f"Model {self.llm_model_name} loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self.llm_tokenizer = None
            self.llm_model = None

    def do_transcribe(self, audio_data, task):
        result, _ = self.get_model().transcribe(
            audio_data,
            language=self.language,
            task=task,
            initial_prompt=self.initial_prompt,
        )
        return result

    def do_process_text(self, transcript):
        if self.device != self.old_device:
            self.do_load_model()

        # Ensure LLM model and tokenizer are initialized
        if not self.llm_tokenizer or not self.llm_model:
            self.logger.error("Tokenizer or model not initialized.")
            return None

        # Tokenize input text for the LLM
        inputs = self.llm_tokenizer(transcript, return_tensors="pt").to(self.device)

        # Generate text using the LLM model
        outputs = self.llm_model.generate(
            **inputs,
            max_new_tokens=100,  # Limit the number of new tokens to generate without truncating input
        )

        # Decode the generated tokens to text
        generated_text = self.llm_tokenizer.decode(
            outputs[0], skip_special_tokens=False
        )  # Keep special tokens for detection

        self.logger.info(f"Generated text using LLM: {generated_text}")

        # Check for the End-of-Text (EOT) token
        eot_token = (
            self.llm_tokenizer.eos_token
        )  # Typically <|endoftext|> for GPT models
        if eot_token in generated_text:
            self.logger.info(f"EOT detected: {eot_token}")
            # Trim the text before the EOT token
            generated_text = generated_text.split(eot_token)[0].strip()
            self.logger.info(f"Trimmed text after EOT detection: {generated_text}")

        # If there is no text left after trimming, return None
        if not generated_text:
            self.logger.info(
                "No valid text generated before EOT. Skipping further processing."
            )
            return None

        # Pass the valid trimmed text to TTS and return the audio buffer
        audio_tensor = self.pipeline.generate(generated_text, lang=self.language)
        audio_np = (audio_tensor.cpu().numpy() * 32767).astype(np.int16)
        if len(audio_np.shape) == 1:
            audio_np = np.expand_dims(
                audio_np, axis=0
            )  # Add a new dimension to make it 2D
        else:
            audio_np = audio_np.T  # Transpose the numpy array if it's not 1D

        duration = len(audio_np) / TTS_SAMPLE_RATE * Gst.SECOND
        buffer = Gst.Buffer.new_wrapped(audio_np.tobytes())

        buffer.pts = Gst.CLOCK_TIME_NONE
        buffer.duration = duration

        return buffer


if CAN_REGISTER_ELEMENT:
    GObject.type_register(WhisperLive)
    __gstelementfactory__ = ("pyml_whisperlive", Gst.Rank.NONE, WhisperLive)
else:
    GlobalLogger().warning(
        "The 'pyml_whisperlive' element will not be registered because required modules were missing."
    )
