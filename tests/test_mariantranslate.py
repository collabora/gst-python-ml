import os
import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
PLUGIN_DIR = BASE_DIR / "plugins" / "python"

os.environ["GST_PLUGIN_PATH"] = str(BASE_DIR / "plugins")
sys.path.insert(0, str(PLUGIN_DIR))

gi = pytest.importorskip("gi")
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

Gst.init(None)

from mariantranslate import (  # noqa: E402
    MarianTranslate,
    _fit_input_ids,
    _input_length_limit,
    _source_input_ids,
    _source_pieces_are_in_vocab,
)

MODEL_NAME = "org/opus-mt-tc-big-en-ko"


class SourceSpm:
    def __init__(self, pieces, encoded):
        self._pieces = pieces
        self._encoded = encoded

    def get_piece_size(self):
        return len(self._pieces)

    def id_to_piece(self, index):
        return self._pieces[index]

    def encode(self, text):
        return list(self._encoded)


class Tokenizer:
    def __init__(self, encoder, pieces, encoded):
        self.encoder = encoder
        self.spm_source = SourceSpm(pieces, encoded)
        self.eos_token_id = 2
        self.calls = []

    def normalize(self, text):
        return text

    def decode(self, ids, skip_special_tokens=False):
        return "translated"

    def __call__(self, text, return_tensors=None, padding=None, **kwargs):
        self.calls.append(text)
        self.kwargs = kwargs
        return {"input_ids": text}


class Model:
    def __init__(self):
        self.inputs = None

    def generate(self, **inputs):
        self.inputs = inputs
        return [[0]]


def test_a_target_side_vocab_uses_source_sentencepiece():
    tokenizer = Tokenizer(
        encoder={"<unk>": 0},
        pieces=["<unk>", "▁hi", "▁world", "▁even"],
        encoded=[1],
    )

    assert _source_pieces_are_in_vocab(tokenizer) is False
    assert _source_input_ids(tokenizer, "hi") == [1, 2]


def test_a_vocab_that_names_the_source_pieces_uses_the_tokenizer():
    # en-fr stores source pieces at ids that are not the SentencePiece indexes.
    tokenizer = Tokenizer(
        encoder={"<unk>": 3, "▁hi": 9, "▁world": 4},
        pieces=["<unk>", "▁hi", "▁world"],
        encoded=[1, 2],
    )

    assert _source_pieces_are_in_vocab(tokenizer) is True


def test_load_uses_the_model_name_it_is_given(monkeypatch):
    import transformers

    names = []
    tokenizer = Tokenizer(
        encoder={"<unk>": 0, "▁hi": 1},
        pieces=["<unk>", "▁hi"],
        encoded=[1],
    )
    model = Model()

    def tokenizer_from_pretrained(name):
        names.append(name)
        return tokenizer

    def model_from_pretrained(name):
        names.append(name)
        return model

    monkeypatch.setattr(
        transformers.MarianTokenizer, "from_pretrained", tokenizer_from_pretrained
    )
    monkeypatch.setattr(
        transformers.MarianMTModel, "from_pretrained", model_from_pretrained
    )

    element = MarianTranslate()
    element.set_property("model-name", MODEL_NAME)
    element.do_load_model()

    assert names == [MODEL_NAME, MODEL_NAME]


def test_load_without_a_model_name_fails():
    element = MarianTranslate()

    with pytest.raises(ValueError, match="model-name"):
        element.do_load_model()


def test_the_element_has_no_language_pair_properties():
    element = MarianTranslate()

    assert element.find_property("model-name") is not None
    assert element.find_property("src") is None
    assert element.find_property("target") is None


def test_a_sepvoc_model_encodes_with_source_sentencepiece(monkeypatch):
    import transformers

    tokenizer = Tokenizer(
        encoder={"<unk>": 0},
        pieces=["<unk>", "▁hi", "▁world", "▁even"],
        encoded=[1],
    )
    model = Model()
    monkeypatch.setattr(
        transformers.MarianTokenizer,
        "from_pretrained",
        lambda name: tokenizer,
    )
    monkeypatch.setattr(
        transformers.MarianMTModel,
        "from_pretrained",
        lambda name: model,
    )

    element = MarianTranslate()
    element.set_property("model-name", MODEL_NAME)
    element.do_load_model()

    assert element.do_translate_text("hi") == "translated"
    assert tokenizer.calls == []
    assert model.inputs["input_ids"].tolist() == [[1, 2]]


def test_a_covered_vocab_encodes_with_the_tokenizer(monkeypatch):
    import transformers

    tokenizer = Tokenizer(
        encoder={"<unk>": 3, "▁hi": 9},
        pieces=["<unk>", "▁hi"],
        encoded=[1],
    )
    model = Model()
    monkeypatch.setattr(
        transformers.MarianTokenizer, "from_pretrained", lambda name: tokenizer
    )
    monkeypatch.setattr(
        transformers.MarianMTModel, "from_pretrained", lambda name: model
    )

    element = MarianTranslate()
    element.set_property("model-name", "org/opus-mt-en-fr")
    element.do_load_model()

    assert element.do_translate_text("hi") == "translated"
    assert tokenizer.calls == ["hi"]


def test_blank_text_is_not_translated():
    element = MarianTranslate()

    assert element.do_translate_text("") == ""
    assert element.do_translate_text(" \n") == ""


def test_source_ids_keep_the_end_marker_when_cut():
    assert _fit_input_ids([1, 2, 3, 4, 5, 9], 4) == [1, 2, 3, 9]
    assert _fit_input_ids([1, 9], 4) == [1, 9]
    assert _fit_input_ids([1, 2, 3], None) == [1, 2, 3]


def test_the_position_table_sets_the_length_limit():
    tokenizer = Tokenizer(encoder={}, pieces=[], encoded=[])
    tokenizer.model_max_length = 512
    model = Model()
    model.config = type("Config", (), {"max_position_embeddings": 1024})()

    assert _input_length_limit(tokenizer, model) == 1024

    bare = Model()
    assert _input_length_limit(tokenizer, bare) == 512
    tokenizer.model_max_length = 10**30
    assert _input_length_limit(tokenizer, bare) is None


def test_a_sepvoc_model_cuts_input_to_the_position_table(monkeypatch):
    import transformers

    tokenizer = Tokenizer(
        encoder={"<unk>": 0},
        pieces=["<unk>", "▁hi", "▁world", "▁even"],
        encoded=[1, 2, 3, 4, 5],
    )
    model = Model()
    model.config = type("Config", (), {"max_position_embeddings": 4})()
    monkeypatch.setattr(
        transformers.MarianTokenizer, "from_pretrained", lambda name: tokenizer
    )
    monkeypatch.setattr(
        transformers.MarianMTModel, "from_pretrained", lambda name: model
    )

    element = MarianTranslate()
    element.set_property("model-name", MODEL_NAME)
    element.do_load_model()

    assert element.do_translate_text("hi " * 20) == "translated"
    assert model.inputs["input_ids"].tolist() == [[1, 2, 3, 2]]


def test_a_covered_vocab_asks_the_tokenizer_to_truncate(monkeypatch):
    import transformers

    tokenizer = Tokenizer(
        encoder={"<unk>": 3, "▁hi": 9},
        pieces=["<unk>", "▁hi"],
        encoded=[1],
    )
    model = Model()
    model.config = type("Config", (), {"max_position_embeddings": 4})()
    monkeypatch.setattr(
        transformers.MarianTokenizer, "from_pretrained", lambda name: tokenizer
    )
    monkeypatch.setattr(
        transformers.MarianMTModel, "from_pretrained", lambda name: model
    )

    element = MarianTranslate()
    element.set_property("model-name", "org/opus-mt-en-fr")
    element.do_load_model()
    element.do_translate_text("hi")

    assert tokenizer.kwargs["truncation"] is True
    assert tokenizer.kwargs["max_length"] == 4
