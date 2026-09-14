SUMMARY = "Runtime stack for gst-python-ml on RZ/V2H (GStreamer 1.24 + Python)"
LICENSE = "MIT"

inherit packagegroup

RDEPENDS:${PN} = " \
    gstreamer1.0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    gstreamer1.0-python \
    python3-core \
    python3-pygobject \
    python3-numpy \
    python3-pycairo \
    python3-opencv \
"
# Notes:
# - gstreamer1.0-python provides the libgstpython.so plugin loader that runs
#   the pyml_* .py elements. It is NOT in the stock AI SDK image.
# - GstAnalytics (used by base_objectdetector / tracker / overlay) ships in
#   gstreamer1.0-plugins-bad once GStreamer is >= 1.24 with the analytics
#   PACKAGECONFIG enabled (see conf/include/gstreamer-1.24.inc).
# - The DRP-AI MERA/TVM *Python* runtime is not a stock Yocto package; install
#   it onto the image separately (see ../README.md "DRP-AI runtime on board").
