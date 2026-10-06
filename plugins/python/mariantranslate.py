# MarianTranslate
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
    from base_translate import BaseTranslate
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_mariantranslate' element will not be available. Element {e}"
    )


# Half coverage splits en-fr vocab ids from en-ko SentencePiece ids.
_SOURCE_PIECE_COVERAGE = 0.5


def _source_pieces_are_in_vocab(tokenizer):
    spm = tokenizer.spm_source
    total = spm.get_piece_size()
    if total == 0:
        return True
    present = sum(spm.id_to_piece(index) in tokenizer.encoder for index in range(total))
    return present / total >= _SOURCE_PIECE_COVERAGE


def _source_input_ids(tokenizer, text):
    ids = tokenizer.spm_source.encode(tokenizer.normalize(text))
    return ids + [tokenizer.eos_token_id]


def _input_length_limit(tokenizer, model):
    config = getattr(model, "config", None)
    positions = getattr(config, "max_position_embeddings", None)
    if isinstance(positions, int) and positions > 0:
        return positions
    declared = getattr(tokenizer, "model_max_length", None)
    if isinstance(declared, int) and 0 < declared < 10**6:
        return declared
    return None


def _fit_input_ids(ids, limit):
    if limit is None or len(ids) <= limit:
        return ids
    if limit == 1:
        return ids[-1:]
    return ids[: limit - 1] + [ids[-1]]


def _on_model_device(inputs, model):
    parameters = getattr(model, "parameters", None)
    if parameters is None:
        return inputs
    try:
        device = next(parameters()).device
    except StopIteration:
        return inputs
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


class MarianTranslate(BaseTranslate):
    __gstmetadata__ = (
        "MarianTranslate",
        "Transform",
        "Processes text using a Large Language Model",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self._encode_with_source_spm = False

    def do_load_model(self):
        from transformers import MarianMTModel, MarianTokenizer

        if not self._model_name:
            raise ValueError("model-name is not set")
        self.tokenizer = MarianTokenizer.from_pretrained(self._model_name)
        self._encode_with_source_spm = not _source_pieces_are_in_vocab(self.tokenizer)
        model = MarianMTModel.from_pretrained(self._model_name)
        if callable(getattr(model, "to", None)):
            model = model.to(self.device)
        if callable(getattr(model, "eval", None)):
            model.eval()
        self.set_model(model)
        self.logger.info(f"Loaded translation model {self._model_name}")

    def do_translate_text(self, text):
        """
        Translates the input text using the MarianMT model.
        """
        if not text or not text.strip():
            return ""
        model = self.get_model()
        if not model or not self.tokenizer:
            self.logger.error("Model or tokenizer is not available.")
            return ""
        limit = _input_length_limit(self.tokenizer, model)
        if self._encode_with_source_spm:
            import torch

            ids = _fit_input_ids(_source_input_ids(self.tokenizer, text), limit)
            input_ids = torch.tensor([ids], dtype=torch.long)
            inputs = {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
        else:
            kwargs = {"return_tensors": "pt", "padding": True}
            if limit is not None:
                kwargs["truncation"] = True
                kwargs["max_length"] = limit
            inputs = self.tokenizer(text, **kwargs)
        inputs = _on_model_device(inputs, model)
        translated = model.generate(**inputs)
        return self.tokenizer.decode(translated[0], skip_special_tokens=True)


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        "pyml_mariantranslate", MarianTranslate
    )
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'pyml_mariantranslate' element will not be registered because required modules are missing."
    )
