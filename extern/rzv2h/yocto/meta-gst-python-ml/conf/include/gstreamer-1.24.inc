# Pin GStreamer to 1.24 across the stack so GstAnalytics is available.
#
# scarthgap's oe-core ships GStreamer 1.22.x. The 1.24 recipes must be present
# in the build (see README: copy the gstreamer1.0* recipes from oe-core
# styhead/master into recipes-multimedia/gstreamer/ of this layer, or layer in
# a newer meta-oe). These PREFERRED_VERSION lines then select them.
#
# require this from local.conf:
#   require ${TOPDIR}/../layers/meta-gst-python-ml/conf/include/gstreamer-1.24.inc

GST_124 ?= "1.24.%"

PREFERRED_VERSION_gstreamer1.0 = "${GST_124}"
PREFERRED_VERSION_gstreamer1.0-plugins-base = "${GST_124}"
PREFERRED_VERSION_gstreamer1.0-plugins-good = "${GST_124}"
PREFERRED_VERSION_gstreamer1.0-plugins-bad = "${GST_124}"
PREFERRED_VERSION_gstreamer1.0-plugins-ugly = "${GST_124}"
PREFERRED_VERSION_gstreamer1.0-libav = "${GST_124}"
PREFERRED_VERSION_gstreamer1.0-python = "${GST_124}"
PREFERRED_VERSION_gstreamer1.0-rtsp-server = "${GST_124}"
PREFERRED_VERSION_gstreamer1.0-vaapi = "${GST_124}"

# GstAnalytics + the object-detection / tracking metas live in -plugins-bad.
# Make sure analytics isn't disabled by a PACKAGECONFIG override.
PACKAGECONFIG:append:pn-gstreamer1.0-plugins-bad = " analytics"
