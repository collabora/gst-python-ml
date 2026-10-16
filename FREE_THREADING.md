# Free-threaded Python on g2g

`g2g-launch-py --threads` runs each hosted element on its own OS thread. On free-threaded CPython 3.14t with the GIL off, those threads run Python at the same time. The gst host does the same with the patched PyGObject and gst-python under [gst host](#gst-host). Released PyGObject turns the GIL back on at import and crashes when it is forced off.

Tested with the yolo detector and pyml_tracker on CUDA, Fedora 43, CPython 3.14.2t. This is experimental. OpenCV has no official free-threaded support, so it is built from source with a patch upstream has not merged.

Run everything from the gst-python-ml checkout, with glass2glass checked out beside it. `PYML_VENV` points `pyml-launch.py` at the free-threaded venv: it re-runs itself there and hands that interpreter, its shared library and its site directories to whichever host it launches.

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
PYML_VENV=$FT_VENV PYML_BACKEND=g2g G2G_LAUNCH=../glass2glass/target-ft/release/g2g-launch-py \
  PYTHONWARNINGS=error::RuntimeWarning python pyml-launch.py \
  filesrc location=data/soccer_tracking.mp4 ! qtdemux ! ffmpegdec ! videoconvertscale \
  ! video/x-raw,width=1280,height=720,format=RGBA \
  ! pyml_yolo model-name=models/football/football device=cuda:0 imgsz=640 confidence=0.1 \
  ! pyml_tracker tracker-type=bytetrack new-track-confidence=0.25 \
  ! fakesink
```

- The embedded interpreter takes its prefix from the first `python` on `PATH`, which the launcher sets. With the repo `.venv` first it dies with `No module named 'encodings'`.
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

## gst host

gst-python's plugin embeds libpython, so the gst host needs a plugin built against `libpython3.14t`, a PyGObject that stays GIL-free, and gst-python override modules that declare themselves GIL-free. Both patched trees are on the `free-threading-python-3.14` branches of [boxerab/pygobject](https://github.com/boxerab/pygobject/tree/free-threading-python-3.14) and [boxerab/gstreamer](https://github.com/boxerab/gstreamer/tree/free-threading-python-3.14). Everything installs into the venv.

```
PYGOBJECT_SRC=$HOME/pygobject-ft
git clone -b free-threading-python-3.14 https://github.com/boxerab/pygobject.git $PYGOBJECT_SRC
uv build --wheel --python $FT_VENV/bin/python --out-dir $PYGOBJECT_SRC/dist $PYGOBJECT_SRC
uv pip install --python $FT_VENV/bin/python $PYGOBJECT_SRC/dist/pygobject-*.whl

GST_SRC=$HOME/gstreamer-ft
git clone --filter=blob:none --sparse --depth=1 --branch free-threading-python-3.14 https://github.com/boxerab/gstreamer.git $GST_SRC
git -C $GST_SRC sparse-checkout set subprojects/gst-python
uvx meson setup $GST_SRC/subprojects/gst-python/build $GST_SRC/subprojects/gst-python \
  --prefix=$FT_VENV -Dpython-exe=$FT_VENV/bin/python -Dplugin=enabled -Dtests=disabled
uvx meson install -C $GST_SRC/subprojects/gst-python/build
```

The system `libgstpython.so` links `libpython3.14.so` and must not load into the 3.14t process. When the venv has its own plugin, the launcher puts that first on `GST_PLUGIN_PATH`, points `GST_PLUGIN_SYSTEM_PATH` at a symlink copy of the system plugin directory without the python plugin, kept under the venv, and gives the venv its own `GST_REGISTRY`.

```
PYML_VENV=$FT_VENV PYTHONWARNINGS=error::RuntimeWarning python pyml-launch.py \
  filesrc location=data/soccer_tracking.mp4 ! qtdemux ! h264parse ! avdec_h264 ! videoconvertscale \
  ! video/x-raw,width=1280,height=720,format=RGBA ! identity eos-after=752 \
  ! pyml_yolo model-name=models/football/football device=cuda:0 imgsz=640 confidence=0.1 \
  ! pyml_tracker tracker-type=bytetrack new-track-confidence=0.25 ! fakesink
```

The same 752 frames into fakesink, pipeline time as `gst-launch-1.0` reports it. The stock row is the same line without `PYML_VENV`, on the system gst-python and PyGObject 3.56.2.

| host | pipeline time | fps |
|---|---|---|
| stock 3.14 | 19.6 s | 38.3 |
| 3.14t, GIL off | 19.1 s | 39.3 |
| 3.14t, `PYTHON_GIL=1` | 19.3 s | 38.9 |

Same conclusion as g2g: yolo sets the frame rate. Before the thread state pin described below the GIL-off run took 32.9 s.

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

With those fixes the gst host ran the yolo pipeline at 23 fps against 38 on stock, and a python element that does nothing cost 1.6 ms a buffer against 0.05 ms. PyGObject enters Python from a streaming thread through `PyGILState_Ensure()` and leaves through `PyGILState_Release()`, which creates a thread state and deletes it again on every call. On 3.14t deleting a thread state costs about 1.7 ms against 0.03 ms on stock, and 3.15.0a5t is the same: `perf` puts the time in mimalloc's abandoned segment collection, `mi_heap_collect_ex` calling `_mi_abandoned_collect` and `mi_segment_try_purge`, with one clock read per segment. The pin patch keeps the first thread state a foreign thread creates until the thread exits, where a `GPrivate` destructor releases it. The empty element then costs 0.04 ms a buffer and the yolo pipeline matches stock. The three reproducers and the yolo pipeline passed repeated runs under `GLIBC_TUNABLES=glibc.malloc.check=3` with the pin.

The installed `libgstpython.so` links `libpython3.14.so.1.0`. A gst host using 3.14t needs this plugin rebuilt against `libpython3.14t.so.1.0`. Loading the regular plugin into the 3.14t process fails during Python thread-state attachment. A rebuilt 1.26.11 plugin loaded under `gst-inspect-1.0` and registered all 48 project elements with the GIL disabled.

The experimental changes and reproducers are preserved here:

- [`pygobject-wrapper-lock.patch`](patches/free-threading/pygobject-wrapper-lock.patch), applied after PyGObject draft commit `5584ef47f975c276fcaf9827ee032214aecb60ba`.
- [`pygobject-thread-state-pin.patch`](patches/free-threading/pygobject-thread-state-pin.patch), applied after the wrapper lock.
- [`gst-python-module-gil.patch`](patches/free-threading/gst-python-module-gil.patch), applied to gst-python 1.26.11.
- [`pygobject_glib_stress.py`](patches/free-threading/pygobject_glib_stress.py), [`pygobject_wrapper_stress.py`](patches/free-threading/pygobject_wrapper_stress.py) and [`pygobject_gstreamer_stress.py`](patches/free-threading/pygobject_gstreamer_stress.py).

The patches are pushed to [boxerab/pygobject](https://github.com/boxerab/pygobject/tree/free-threading-python-3.14) and [boxerab/gstreamer](https://github.com/boxerab/gstreamer/tree/free-threading-python-3.14) on branches named `free-threading-python-3.14`.

The remaining work is:

1. Replace or validate the broad PyGObject wrapper lock and audit the remaining wrapper reads.
2. Add the wrapper identity test and both cache stress tests to PyGObject.
3. Audit the gst-python override modules beyond their module declarations.
4. Run `tests/test_pipelines.py` with `PYML_VENV=$FT_VENV` on both backends.
5. Upstream the thread state pin, or get CPython to make thread state deletion cheap.

## Not working yet

- `track=true` on the yolo element: the ultralytics trackers import `lap`, which turns the GIL back on. Use pyml_tracker.
- gst-only elements, the ones that subclass `GstBase` or `Gst.Element` such as pyml_vad and pyml_clap, don't run on g2g.
- Two elements can call cv2 at the same moment. The author of opencv#30134 stress-tested 8 threads on `resize`, `cvtColor`, `warpPerspective` and drawing calls. pyml_tracker's `estimateAffinePartial2D` was not in that set.
