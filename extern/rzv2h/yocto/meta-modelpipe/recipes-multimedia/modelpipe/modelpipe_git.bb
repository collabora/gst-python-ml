SUMMARY = "ModelPipe elements (mp_*) + DRP-AI engine for RZ/V2H"
DESCRIPTION = "Installs the pure-Python GStreamer elements and sets GST_PLUGIN_PATH/PYTHONPATH."
LICENSE = "LGPL-2.1-or-later"
LIC_FILES_CHKSUM = "file://COPYING;md5=<FILL_IN>"

# Point this at your ModelPipe source. Examples:
#   SRC_URI = "git://github.com/collabora/modelpipe.git;branch=main;protocol=https"
#   SRCREV  = "<commit>"
# or a local checkout via:  SRC_URI = "file:///path/to/modelpipe"
SRC_URI = "git://github.com/collabora/modelpipe.git;branch=master;protocol=https"
SRCREV = "${AUTOREV}"
S = "${WORKDIR}/git"

# Pure-Python elements: nothing to compile.
do_compile[noexec] = "1"

MODELPIPE_DIR = "${datadir}/modelpipe"

do_install() {
    install -d ${D}${MODELPIPE_DIR}
    cp -r ${S}/plugins ${D}${MODELPIPE_DIR}/plugins

    # Environment so GStreamer finds the .py elements and Python finds the pkg.
    install -d ${D}${sysconfdir}/profile.d
    cat > ${D}${sysconfdir}/profile.d/modelpipe.sh <<EOF
export GST_PLUGIN_PATH="${MODELPIPE_DIR}/plugins:\$GST_PLUGIN_PATH"
export PYTHONPATH="${MODELPIPE_DIR}/plugins/python:\$PYTHONPATH"
EOF
}

FILES:${PN} = "${MODELPIPE_DIR} ${sysconfdir}/profile.d/modelpipe.sh"

# Runtime deps: the GStreamer/Python stack (see packagegroup) + the DRP-AI
# Python runtime (provided out-of-band — see ../README.md).
RDEPENDS:${PN} = "packagegroup-modelpipe"
