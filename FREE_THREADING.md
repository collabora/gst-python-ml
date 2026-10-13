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

## Not working yet

- `track=true` on the yolo element: the ultralytics trackers import `lap`, which turns the GIL back on. Use pyml_tracker.
- gst-only elements, the ones that subclass `GstBase` or `Gst.Element` such as pyml_vad and pyml_clap, don't run on g2g.
- Two elements can call cv2 at the same moment. The author of opencv#30134 stress-tested 8 threads on `resize`, `cvtColor`, `warpPerspective` and drawing calls. pyml_tracker's `estimateAffinePartial2D` was not in that set.
