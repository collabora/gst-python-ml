SUMMARY = "gst-python-ml elements (pyml_*) + DRP-AI engine for RZ/V2H"
DESCRIPTION = "Installs the pure-Python GStreamer elements and sets GST_PLUGIN_PATH/PYTHONPATH."
LICENSE = "LGPL-2.1-or-later"
LIC_FILES_CHKSUM = "file://COPYING;md5=<FILL_IN>"

# Point this at your gst-python-ml source. Examples:
#   SRC_URI = "git://github.com/collabora/gst-python-ml.git;branch=main;protocol=https"
#   SRCREV  = "<commit>"
# or a local checkout via:  SRC_URI = "file:///path/to/gst-python-ml"
SRC_URI = "git://github.com/collabora/gst-python-ml.git;branch=master;protocol=https"
SRCREV = "${AUTOREV}"
S = "${WORKDIR}/git"

# Pure-Python elements: nothing to compile.
do_compile[noexec] = "1"

PYML_DIR = "${datadir}/gst-python-ml"

do_install() {
    install -d ${D}${PYML_DIR}
    cp -r ${S}/plugins ${D}${PYML_DIR}/plugins

    # Environment so GStreamer finds the .py elements and Python finds the pkg.
    install -d ${D}${sysconfdir}/profile.d
    cat > ${D}${sysconfdir}/profile.d/gst-python-ml.sh <<EOF
export GST_PLUGIN_PATH="${PYML_DIR}/plugins:\$GST_PLUGIN_PATH"
export PYTHONPATH="${PYML_DIR}/plugins/python:\$PYTHONPATH"
EOF
}

FILES:${PN} = "${PYML_DIR} ${sysconfdir}/profile.d/gst-python-ml.sh"

# Runtime deps: the GStreamer/Python stack (see packagegroup) + the DRP-AI
# Python runtime (provided out-of-band — see ../README.md).
RDEPENDS:${PN} = "packagegroup-gst-python-ml"
