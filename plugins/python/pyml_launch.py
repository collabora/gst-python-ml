# pyml-launch
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

"""Run one pipeline description on whichever backend `GSTML_BACKEND` selects.

`gst` hands the pipeline to `gst-launch-1.0` unchanged. `g2g` rewrites it for
`g2g-launch-py`, which spells three things differently:

  * a `pyml_*` element is the generic `pyelement` host plus the module and class
    to load;
  * `pyml_overlay` is g2g's own `analyticsoverlay`, a native element with its own
    properties rather than a hosted one;
  * `pyelement` and `analyticsoverlay` both work on RGBA, so a raw-video caps
    filter that leaves the format open has to pin it.

Everything else is named the same on both: `filesrc`, `decodebin`,
`videoconvert`, `videoscale`, `autovideosink`. Write the pipeline the way the
README does, in `gst-launch` spelling, and this translates it.
"""

import ast
import os
import shutil
import site
import subprocess
import sys
from pathlib import Path

#: The checkout's `plugins` tree, when this module was loaded from one. An
#: installed copy sits in site-packages, which carries no element modules.
_PLUGINS = Path(__file__).resolve().parent.parent
CHECKOUT_PLUGINS = _PLUGINS if _PLUGINS.name == "plugins" else None

USAGE = "usage: pyml-launch <element> [key=value ...] ! <element> ! ..."

#: The g2g launcher, and the checkout it is built in, looked for beside this one.
G2G_LAUNCH_NAME = "g2g-launch-py"
G2G_CHECKOUT_NAME = "glass2glass"

#: gst-python-ml elements that g2g implements natively instead of hosting.
NATIVE_EQUIVALENTS = {"pyml_overlay": "analyticsoverlay"}

#: `pyelement` hosts an RGBA frame by default and `analyticsoverlay` draws on
#: RGBA8 only, so a caps filter that names no format would fail to negotiate.
G2G_RAW_VIDEO_FORMAT = "RGBA"


def plugin_dir():
    """The directory holding the element modules: the checkout this module came
    from, or the `python` subdirectory of a `GST_PLUGIN_PATH` entry."""
    if CHECKOUT_PLUGINS:
        return CHECKOUT_PLUGINS / "python"
    for entry in os.environ.get("GST_PLUGIN_PATH", "").split(os.pathsep):
        candidate = Path(entry) / "python"
        if candidate.is_dir():
            return candidate
    raise SystemExit(
        "pyml-launch: no gst-python-ml element modules found; point "
        "GST_PLUGIN_PATH at the checkout's plugins directory"
    )


def element_shells(directory=None):
    """Map each element name to the `(module, class)` implementing it.

    Read out of the sources with `ast` rather than by importing: a plugin pulls
    in torch and friends, and every one whose dependencies are missing would
    drop out of the map.
    """
    shells = {}
    for path in sorted((directory or plugin_dir()).glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = node.func
            name = (
                called.attr
                if isinstance(called, ast.Attribute)
                else getattr(called, "id", None)
            )
            if name != "register_gst_element" or len(node.args) < 2:
                continue
            element, cls = node.args[0], node.args[1]
            if isinstance(element, ast.Constant) and isinstance(cls, ast.Name):
                shells[element.value] = (path.stem, cls.id)
    return shells


def rewrite_for_g2g(pipeline, shells):
    """Translate a `gst-launch` pipeline into its `g2g-launch-py` spelling."""
    return " ! ".join(
        rewrite_segment(segment.strip(), shells) for segment in pipeline.split("!")
    )


def rewrite_segment(segment, shells):
    if not segment:
        return segment
    parts = segment.split(None, 1)
    head = parts[0]
    properties = parts[1] if len(parts) > 1 else ""

    # A caps filter is the one segment whose first token is a media type; no
    # element name contains a slash.
    if "/" in head:
        return with_pinned_format(segment)

    native = NATIVE_EQUIVALENTS.get(head)
    if native:
        if properties:
            raise SystemExit(
                f"pyml-launch: {head} properties ({properties}) have no equivalent on "
                f"g2g's {native}; it takes show-label, show-track, show-score, "
                f"show-trail, trail-length, thickness and mask-alpha instead"
            )
        return native

    shell = shells.get(head)
    if shell:
        module, cls = shell
        hosted = f"pyelement module={module} class={cls}"
        return f"{hosted} {properties}" if properties else hosted

    if head.startswith("pyml_"):
        raise SystemExit(f"pyml-launch: no gst-python-ml element named {head!r}")

    return segment


def with_pinned_format(caps):
    if not caps.startswith("video/x-raw") or "format=" in caps:
        return caps
    return f"{caps},format={G2G_RAW_VIDEO_FORMAT}"


def _prepend_path(existing, *entries):
    paths = [str(entry) for entry in entries]
    if existing:
        paths.append(existing)
    return os.pathsep.join(paths)


def launch_environment():
    """The environment both launchers need.

    Neither runs under this interpreter: GStreamer's Python loader embeds the
    system one and g2g embeds its own, so torch and the rest are only reachable
    if the venv's site directories are on `PYTHONPATH`.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = _prepend_path(
        env.get("PYTHONPATH"), plugin_dir(), *site.getsitepackages()
    )
    return env


def gst_command(argv):
    env = launch_environment()
    if CHECKOUT_PLUGINS:
        env["GST_PLUGIN_PATH"] = _prepend_path(
            env.get("GST_PLUGIN_PATH"), CHECKOUT_PLUGINS
        )
    return ["gst-launch-1.0", *argv], env


def g2g_binary():
    """The `g2g-launch-py` to run: an explicit `G2G_LAUNCH`, else the release
    build in the glass2glass checkout, else whatever is on `PATH`."""
    explicit = os.environ.get("G2G_LAUNCH")
    if explicit:
        return explicit
    checkout = os.environ.get("G2G_DIR")
    if not checkout and CHECKOUT_PLUGINS:
        checkout = CHECKOUT_PLUGINS.parent.parent / G2G_CHECKOUT_NAME
    if checkout:
        release = Path(checkout) / "target" / "release" / G2G_LAUNCH_NAME
        if release.is_file():
            return str(release)
    return shutil.which(G2G_LAUNCH_NAME)


def g2g_command(pipeline):
    binary = g2g_binary()
    if not binary:
        raise SystemExit(
            "pyml-launch: no g2g-launch-py release build found; point G2G_LAUNCH "
            "at one, or G2G_DIR at a glass2glass checkout, and build it with: "
            "PYO3_PYTHON=$(which python) cargo build --release -p g2g-python "
            "--features ml --bin g2g-launch-py"
        )
    return [binary, *pipeline.split()], launch_environment()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv:
        raise SystemExit(USAGE)

    backend = os.environ.get("GSTML_BACKEND", "gst").lower()
    if backend == "gst":
        command, env = gst_command(argv)
    elif backend == "g2g":
        pipeline = rewrite_for_g2g(" ".join(argv), element_shells())
        print(f"pyml-launch: {pipeline}", file=sys.stderr)
        command, env = g2g_command(pipeline)
    else:
        raise SystemExit(
            f"pyml-launch: unknown GSTML_BACKEND={backend!r}; use 'gst' or 'g2g'"
        )

    return subprocess.call(command, env=env)


if __name__ == "__main__":
    sys.exit(main())
