# Free-threaded Python on g2g

`g2g-launch-py --threads` runs each hosted element on its own OS thread. On free-threaded CPython 3.14t with the GIL off, those threads run Python at the same time. The gst host can't do this: pygobject turns the GIL back on at import and crashes when it is forced off.

Tested with the yolo detector and pyml_tracker on CUDA, Fedora 43, CPython 3.14.2t. This is experimental. OpenCV has no official free-threaded support, so it is built from source with a patch upstream has not merged.

Run everything from the gst-python-ml checkout, with glass2glass checked out beside it.

## Venv

```
FT_VENV=$HOME/ft314
uv python install 3.14t
uv venv --python 3.14t $FT_VENV
uv pip install --python $FT_VENV/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  torch torchvision numpy scipy matplotlib pillow pyyaml requests psutil ultralytics-thop
uv pip uninstall --python $FT_VENV/bin/python triton
uv pip install --python $FT_VENV/bin/python --no-deps ultralytics
```

- triton turns the GIL back on at import, and torchvision imports it. torch needs triton only for `torch.compile`.
- `--no-deps` skips polars, which ships only abi3 wheels. ultralytics imports polars only for training and plots.

## OpenCV

opencv-python ships only abi3 wheels, which a free-threaded interpreter can't load. The build drops the limited API ([opencv-python#1051](https://github.com/opencv/opencv-python/pull/1051)) and declares the cv2 module GIL-free ([opencv#30134](https://github.com/opencv/opencv/pull/30134)). It takes about 10 minutes on 16 threads.

```
OPENCV_DIR=$HOME/opencv-ft
mkdir -p $OPENCV_DIR
curl -L -o $OPENCV_DIR/opencv.tar.gz https://files.pythonhosted.org/packages/source/o/opencv-python-headless/opencv_python_headless-5.0.0.93.tar.gz
tar xzf $OPENCV_DIR/opencv.tar.gz -C $OPENCV_DIR
OPENCV_SRC=$OPENCV_DIR/opencv_python_headless-5.0.0.93
sed -i '/"-DPYTHON3_LIMITED_API=ON",/d' $OPENCV_SRC/setup.py
sed -i 's|^    PyObject\* m = PyModule_Create(&cv2_moduledef);$|&\n#ifdef Py_GIL_DISABLED\n    PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED);\n#endif|' \
  $OPENCV_SRC/opencv/modules/python/src2/cv2.cpp
CMAKE_BUILD_PARALLEL_LEVEL=10 uv build --wheel --python $FT_VENV/bin/python --out-dir $OPENCV_DIR/wheels $OPENCV_SRC
uv pip install --python $FT_VENV/bin/python $OPENCV_DIR/wheels/opencv_python_headless-*-cp314t-*.whl
```

## Launcher

```
(cd ../glass2glass && PYO3_PYTHON=$FT_VENV/bin/python CARGO_TARGET_DIR=target-ft \
  cargo build --release -p g2g-python --bin g2g-launch-py --features launch,g2g-plugins/linux-full)
```

pyo3 links the libpython of `PYO3_PYTHON`. The separate target dir keeps the stock 3.14 build in `target/`.

## Run

The weights come from `python demo/football/fetch_models.py pt` in the repo venv.

```
FT_LIB=$(dirname $(uv python find 3.14t))/../lib
PATH=$FT_VENV/bin:$PATH PYTHONPATH=$PWD/plugins/python LD_LIBRARY_PATH=$FT_LIB \
  PYTHONWARNINGS=error::RuntimeWarning \
  ../glass2glass/target-ft/release/g2g-launch-py --threads \
  filesrc location=data/soccer_tracking.mp4 ! qtdemux ! ffmpegdec ! videoconvertscale \
  ! video/x-raw,width=1280,height=720,format=RGBA \
  ! pyelement module=yolo class=YOLOTransform model-name=models/football/football device=cuda:0 imgsz=640 confidence=0.1 \
  ! pyelement module=tracker class=TrackerTransform tracker-type=bytetrack new-track-confidence=0.25 \
  ! fakesink
```

- Call the launcher directly. `pyml-launch.py` re-runs itself under the repo `.venv`.
- The embedded interpreter takes its prefix from the first `python` on `PATH`. With the repo `.venv` first it dies with `No module named 'encodings'`.
- `PYTHONPATH` must not keep the repo venv's site dir. Its numpy fails to load under 3.14t.
- `PYTHONWARNINGS=error::RuntimeWarning` fails the element at import if any module turns the GIL back on, instead of printing a warning.

## Measurements

The first 15 s of `data/soccer_tracking.mp4` (752 frames) into fakesink, glass2glass at d949f2d5:

| host | wall | fps |
|---|---|---|
| stock 3.14 | 20.7 s | 36.3 |
| 3.14t, GIL off | 20.0 s | 37.7 |
| 3.14t, `PYTHON_GIL=1` | 20.1 s | 37.5 |

Free threading doesn't speed this pipeline up. yolo spends about 22 ms per frame in Python and sets the frame rate, and the GIL is not what holds it back. Before d949f2d5 each hosted element built and tore down a Python thread state every frame, which cost 5 ms a frame on 3.14t.

## PyGObject investigation

Tested on Fedora 43 with CPython 3.14.2t, GStreamer 1.26.11 and PyGObject 3.58.1. PyPI distributes PyGObject as source only, so a `cp314t` build succeeds. The released `_gi` module does not declare free-threading support. CPython therefore enables the GIL when `_gi` is imported. `PYTHON_GIL=0` overrides this protection and exposes data races.

A 12-thread reproducer calling `GLib.filename_from_uri()` 200,000 times per thread failed within seconds with `malloc(): unaligned tcache chunk detected`. Two unlocked PyGObject caches independently caused corruption:

- `free_arg_state[10]` in `gi/pygi-invoke.c` reuses native argument blocks.
- `free_list[10]` and `numfree[10]` in `gi/pygi-resulttuple.c` reuse `ResultTuple` objects.

Disabling only one cache still crashed. Disabling both caches completed five runs and 12 million calls. This matches the changes in the upstream [free-threading draft](https://gitlab.gnome.org/GNOME/pygobject/-/merge_requests/565), which tracks the remaining work in [issue 646](https://gitlab.gnome.org/GNOME/pygobject/-/work_items/646).

A `cp314t` wheel built from the upstream draft kept the GIL disabled without `PYTHON_GIL=0`. It completed the 12 million call test and 400,000 callbacks from eight simultaneous GStreamer pipelines when the gst-python overrides were absent.

The draft still races when it creates Python wrappers for GObjects. `pygobject_new_full()` reads the wrapper from GObject qdata, creates one when absent, then stores it. Concurrent first access can create two wrappers for one GObject. A 32-thread test calling `get_static_pad("src")` returned two wrappers in 5 of 10 runs. `pygobject_clear()` can then remove the qdata entry for the other wrapper.

An experimental recursive lock around wrapper lookup, registration and removal returned one wrapper in 20 of 20 runs. The lock also removed the allocator corruption seen when the gst-python overrides were enabled. Five runs of eight simultaneous pipelines completed two million callbacks with the GIL disabled. The lock is deliberately broad and still needs upstream review for callback and destruction behavior.

gst-python also needs free-threading work. `_gi_gst` and `_gi_gst_analytics` use single-phase module initialization without calling `PyUnstable_Module_SetGIL()`. A matching 1.26.11 build with that declaration imported with the GIL disabled. With the PyGObject wrapper lock, its overrides passed the concurrent pipeline test.

The installed `libgstpython.so` links `libpython3.14.so.1.0`. A gst host using 3.14t needs this plugin rebuilt against `libpython3.14t.so.1.0`. Loading the regular plugin into the 3.14t process fails during Python thread-state attachment. A rebuilt 1.26.11 plugin loaded under `gst-inspect-1.0` and registered all 48 project elements with the GIL disabled.

The experimental changes and reproducers are preserved here:

- [`pygobject-wrapper-lock.patch`](patches/free-threading/pygobject-wrapper-lock.patch), applied after PyGObject draft commit `5584ef47f975c276fcaf9827ee032214aecb60ba`.
- [`gst-python-module-gil.patch`](patches/free-threading/gst-python-module-gil.patch), applied to gst-python 1.26.11.
- [`pygobject_glib_stress.py`](patches/free-threading/pygobject_glib_stress.py), [`pygobject_wrapper_stress.py`](patches/free-threading/pygobject_wrapper_stress.py) and [`pygobject_gstreamer_stress.py`](patches/free-threading/pygobject_gstreamer_stress.py).

The remote forks are [boxerab/pygobject](https://github.com/boxerab/pygobject) and [boxerab/gstreamer](https://github.com/boxerab/gstreamer). Patch branches have not been pushed.

The remaining work is:

1. Replace or validate the broad PyGObject wrapper lock and audit the remaining wrapper reads.
2. Add the wrapper identity test and both cache stress tests to PyGObject.
3. Audit the gst-python override modules beyond their module declarations.
4. Run the gst host and project pipelines repeatedly with allocator diagnostics.

## Not working yet

- `track=true` on the yolo element: the ultralytics trackers import `lap`, which turns the GIL back on. Use pyml_tracker.
- gst-only elements, the ones that subclass `GstBase` or `Gst.Element` such as pyml_vad and pyml_clap, don't run on g2g.
- Two elements can call cv2 at the same moment. The author of opencv#30134 stress-tested 8 threads on `resize`, `cvtColor`, `warpPerspective` and drawing calls. pyml_tracker's `estimateAffinePartial2D` was not in that set.
