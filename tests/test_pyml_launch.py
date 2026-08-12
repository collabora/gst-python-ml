"""Unit tests for `pyml-launch`'s gst -> g2g pipeline rewrite.

These run the real `element_shells` scan over the real plugin directory, so a
plugin that stops declaring `register_gst_element` fails a test rather than
silently dropping out of the map.
"""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "plugins" / "python"))

import pyml_launch  # noqa: E402


@pytest.fixture(scope="module")
def shells():
    return pyml_launch.element_shells()


def test_element_shells_finds_the_hosted_elements(shells):
    assert shells["pyml_yolo"] == pyml_launch.ElementShell("yolo", "YOLOTransform")
    assert shells["pyml_objectdetector"].module == "objectdetector"
    assert shells["pyml_overlay"] == pyml_launch.ElementShell("overlay", "Overlay")


def test_an_element_several_chains_feed_becomes_the_batching_host(shells):
    pipeline = (
        "videotestsrc ! pyml_streammux name=mux "
        "videotestsrc ! mux. videotestsrc ! mux. "
        "mux. ! fakesink"
    ).split()
    rewritten = pyml_launch.rewrite_for_g2g(pipeline, shells)
    assert "pyaggregator" in rewritten
    assert "pyelement" not in rewritten


def test_a_single_input_element_stays_on_the_one_in_host(shells):
    # Whisper derives from GstBase.Aggregator but takes one chain, which g2g
    # hosts as a transform.
    assert (
        pyml_launch.rewrite_segment("pyml_whispertranscribe language=ko", shells)[0]
        == "pyelement"
    )


def test_element_shells_resolves_a_name_given_as_a_class_constant(shells):
    # kafkasink registers under `KafkaSink.GST_PLUGIN_NAME`, not a literal.
    assert shells["pyml_kafkasink"] == pyml_launch.ElementShell(
        "kafkasink", "KafkaSink"
    )


def test_hosted_element_becomes_pyelement_keeping_its_properties(shells):
    assert pyml_launch.rewrite_segment(
        "pyml_yolo model-name=yolo11m device=cuda:0", shells
    ) == [
        "pyelement",
        "module=yolo",
        "class=YOLOTransform",
        "model-name=yolo11m",
        "device=cuda:0",
    ]


def test_overlay_becomes_the_native_element(shells):
    assert pyml_launch.rewrite_segment("pyml_overlay", shells) == ["analyticsoverlay"]


def test_overlay_properties_carry_over_under_the_native_name(shells):
    assert pyml_launch.rewrite_segment("pyml_overlay tracking=True", shells) == [
        "analyticsoverlay",
        "show-track=True",
    ]


def test_an_overlay_property_with_no_counterpart_is_refused(shells):
    with pytest.raises(SystemExit, match="meta-path"):
        pyml_launch.rewrite_segment(
            "pyml_overlay meta-path=data/sample_metadata.json", shells
        )


def test_unknown_pyml_element_is_refused(shells):
    with pytest.raises(SystemExit, match="pyml_nonesuch"):
        pyml_launch.rewrite_segment("pyml_nonesuch", shells)


def test_shared_elements_pass_through_untouched(shells):
    for segment in ("filesrc location=data/people.mp4", "decodebin", "videoscale"):
        assert pyml_launch.rewrite_segment(segment, shells) == segment.split()


def test_raw_video_caps_gain_a_format_only_when_they_lack_one(shells):
    assert pyml_launch.rewrite_segment("video/x-raw,width=640,height=480", shells) == [
        "video/x-raw,width=640,height=480,format=RGBA"
    ]
    pinned = "video/x-raw,format=NV12,width=640,height=480"
    assert pyml_launch.rewrite_segment(pinned, shells) == [pinned]


def test_caps_written_with_spaces_become_one_token(shells):
    assert pyml_launch.rewrite_segment(
        ["video/x-raw,", "width=320,", "height=240"], shells
    ) == ["video/x-raw,width=320,height=240,format=RGBA"]


def test_a_property_value_with_spaces_keeps_them(shells):
    assert pyml_launch.rewrite_segment(
        ["pyml_clip", "labels=person, bicycle, car", "top-k=3"], shells
    ) == [
        "pyelement",
        "module=clip",
        "class=CLIPTransform",
        'labels="person, bicycle, car"',
        "top-k=3",
    ]


def test_a_property_value_keeps_the_quotes_inside_it(shells):
    assert (
        pyml_launch.rewrite_segment(["pyml_alert", 'rules={"class":"person"}'], shells)[
            -1
        ]
        == r"rules={\"class\":\"person\"}"
    )


def test_a_sink_drops_the_clock_properties_g2g_has_no_counterpart_for(shells):
    assert pyml_launch.rewrite_segment("fakesink async=0 sync=0", shells) == [
        "fakesink"
    ]


def test_a_sink_asked_to_wait_on_the_clock_is_refused(shells):
    with pytest.raises(SystemExit, match="clocksync"):
        pyml_launch.rewrite_segment("autovideosink sync=true", shells)


def test_a_hosted_element_named_like_a_sink_keeps_its_own_properties(shells):
    # `sync` on a g2g sink names a clock wait that does not exist there, but on
    # `pyml_kafkasink` it is the Kafka producer's own knob.
    assert pyml_launch.rewrite_segment("pyml_kafkasink sync=true", shells) == [
        "pyelement",
        "module=kafkasink",
        "class=KafkaSink",
        "sync=true",
    ]


def test_textoverlay_drops_the_wait_it_never_does(shells):
    assert pyml_launch.rewrite_segment(
        "textoverlay name=overlay wait-text=false", shells
    ) == ["textoverlay", "name=overlay"]


def test_whole_readme_pipeline_rewrites(shells):
    pipeline = (
        "filesrc location=data/soccer_tracking.mp4 ! decodebin ! videoconvert "
        "! videoscale ! video/x-raw,width=640,height=480 "
        "! pyml_yolo model-name=yolo11m device=cuda:0 track=True "
        "! pyml_overlay ! videoconvert ! autovideosink"
    )
    assert (
        pyml_launch.rewrite_for_g2g(pipeline.split(), shells)
        == (
            "filesrc location=data/soccer_tracking.mp4 ! decodebin ! videoconvert "
            "! videoscale ! video/x-raw,width=640,height=480,format=RGBA "
            "! pyelement module=yolo class=YOLOTransform model-name=yolo11m "
            "device=cuda:0 track=True "
            "! analyticsoverlay ! videoconvert ! autovideosink"
        ).split()
    )


def test_hosted_element_takes_the_format_the_caps_ahead_of_it_pin(shells):
    pipeline = (
        "videoconvert ! video/x-raw,format=RGB,width=640,height=640 "
        "! pyml_inference engine-name=onnx"
    ).split()
    assert pyml_launch.rewrite_for_g2g(pipeline, shells)[-4:] == [
        "module=inference",
        "class=GenericInferenceTransform",
        "format=RGB",
        "engine-name=onnx",
    ]


def test_the_batching_host_takes_the_caps_of_an_audio_in_text_out_element(shells):
    assert pyml_launch.rewrite_segment(
        "pyml_whispertranscribe", shells, host=pyml_launch.PY_AGGREGATOR
    ) == [
        "pyaggregator",
        "module=whispertranscribe",
        "class=WhisperTranscribe",
        "input-caps=audio/x-raw,format=S16LE,layout=interleaved,rate=16000,channels=1",
        "output-caps=text/x-raw,format=utf8",
    ]


def test_the_batching_host_takes_the_caps_of_a_text_in_audio_out_element(shells):
    assert pyml_launch.rewrite_segment(
        "pyml_coquitts", shells, host=pyml_launch.PY_AGGREGATOR
    ) == [
        "pyaggregator",
        "module=coquitts",
        "class=CoquiTTS",
        "input-caps=text/x-raw,format=utf8",
        "output-caps=audio/x-raw,format=S16LE,layout=interleaved,rate=22050,channels=1",
    ]


def test_the_one_in_host_takes_the_caps_of_an_audio_in_text_out_element(shells):
    pipeline = (
        "filesrc location=data/audio_sample.wav ! decodebin ! audioconvert "
        "! pyml_whispertranscribe language=ko ! fakesink"
    ).split()
    assert (
        pyml_launch.rewrite_for_g2g(pipeline, shells)
        == (
            "filesrc location=data/audio_sample.wav ! decodebin ! audioconvert "
            "! pyelement module=whispertranscribe class=WhisperTranscribe "
            "input-caps=audio/x-raw,format=S16LE,layout=interleaved,rate=16000,channels=1 "
            "output-caps=text/x-raw,format=utf8 language=ko "
            "! fakesink"
        ).split()
    )


def test_the_one_in_host_takes_the_caps_of_a_text_in_audio_out_element(shells):
    pipeline = "filesrc location=data/lines.txt ! pyml_coquitts ! fakesink".split()
    assert (
        pyml_launch.rewrite_for_g2g(pipeline, shells)
        == (
            "filesrc location=data/lines.txt "
            "! pyelement module=coquitts class=CoquiTTS "
            "input-caps=text/x-raw,format=utf8 "
            "output-caps=audio/x-raw,format=S16LE,layout=interleaved,rate=22050,channels=1 "
            "! fakesink"
        ).split()
    )


def test_a_leaf_that_restates_a_rate_wins_over_the_base_it_inherits_from(shells):
    # Demucs runs at the base class's 44100, Sepformer restates both pads at 8000.
    assert shells["pyml_demucs"].caps == (
        (
            "input-caps",
            "audio/x-raw,format=S16LE,layout=interleaved,rate=44100,channels=1",
        ),
        (
            "output-caps",
            "audio/x-raw,format=S16LE,layout=interleaved,rate=44100,channels=1",
        ),
    )
    assert shells["pyml_sepformer"].caps == (
        (
            "input-caps",
            "audio/x-raw,format=S16LE,layout=interleaved,rate=8000,channels=1",
        ),
        (
            "output-caps",
            "audio/x-raw,format=S16LE,layout=interleaved,rate=8000,channels=1",
        ),
    )


def test_a_leaf_inherits_the_pad_it_does_not_restate(shells):
    # WhisperLive transcribes into speech, so only its src pad differs.
    assert shells["pyml_whisperlive"].caps == (
        (
            "input-caps",
            "audio/x-raw,format=S16LE,layout=interleaved,rate=16000,channels=1",
        ),
        (
            "output-caps",
            "audio/x-raw,format=S16LE,layout=interleaved,rate=24000,channels=1",
        ),
    )


def test_an_element_declaring_no_caps_gets_no_caps_properties(shells):
    pipeline = (
        "videotestsrc ! pyml_streammux name=mux "
        "videotestsrc ! mux. videotestsrc ! mux. "
        "mux. ! fakesink"
    ).split()
    rewritten = pyml_launch.rewrite_for_g2g(pipeline, shells)
    assert "pyaggregator" in rewritten
    assert not [token for token in rewritten if token.startswith(("input-", "output-"))]


def test_the_caps_survive_the_rewrite_of_a_whole_pipeline(shells):
    pipeline = (
        "filesrc location=data/audio_sample.wav ! decodebin ! audioconvert "
        "! pyml_whispertranscribe name=stt language=ko "
        "audiotestsrc ! stt. audiotestsrc ! stt. stt. ! fakesink"
    ).split()
    assert (
        pyml_launch.rewrite_for_g2g(pipeline, shells)
        == (
            "filesrc location=data/audio_sample.wav ! decodebin ! audioconvert "
            "! pyaggregator module=whispertranscribe class=WhisperTranscribe "
            "input-caps=audio/x-raw,format=S16LE,layout=interleaved,rate=16000,channels=1 "
            "output-caps=text/x-raw,format=utf8 name=stt language=ko "
            "audiotestsrc ! stt. audiotestsrc ! stt. stt. ! fakesink"
        ).split()
    )


def g2g_pipeline(*segments):
    """Run a pipeline on the g2g launcher, returning what it printed.

    Skips where there is no build to check against, so the drift check runs for
    whoever has both repos and never blocks whoever has one.
    """
    binary = pyml_launch.g2g_binary()
    if not binary:
        pytest.skip("needs a g2g-launch-py build to check the names against")
    return subprocess.run(
        [binary, *segments],
        capture_output=True,
        text=True,
        timeout=120,
    )


#: Enough of a pipeline to negotiate raw video into the element under test.
G2G_VIDEO_SOURCE = (
    "videotestsrc",
    "num-buffers=1",
    "!",
    "videoconvert",
    "!",
    f"video/x-raw,format={pyml_launch.G2G_RAW_VIDEO_FORMAT}",
    "!",
)


def test_the_native_element_and_the_names_it_renames_to_still_exist():
    """`NATIVE_EQUIVALENTS` and `NATIVE_PROPERTIES` copy names out of glass2glass,
    and `G2G_RAW_VIDEO_FORMAT` states what that element negotiates. Nothing here
    notices when any of the three changes there, so run one and see."""
    for element, native in pyml_launch.NATIVE_EQUIVALENTS.items():
        renames = pyml_launch.NATIVE_PROPERTIES.get(element, {})
        properties = [f"{name}=true" for name in renames.values()]
        result = g2g_pipeline(*G2G_VIDEO_SOURCE, native, *properties, "!", "fakesink")
        assert result.returncode == 0, (
            f"{element} rewrites to `{native} {' '.join(properties)}`, which g2g "
            f"no longer runs:\n{result.stdout}{result.stderr}"
        )


def test_an_element_read_from_twice_is_not_a_muxer(shells):
    # Two chains start at `cap.`, reading two source pads, which is a fan-out.
    pipeline = (
        "videotestsrc ! pyml_caption_qwen name=cap "
        "cap.src ! fakesink cap.text_src ! fakesink"
    ).split()
    rewritten = pyml_launch.rewrite_for_g2g(pipeline, shells)
    assert "pyelement" in rewritten
    assert "pyaggregator" not in rewritten
