# Framework-primitive shims (g2g backend)
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.

"""Stand-ins for the GStreamer framework primitives leaf elements reference.

The g2g host drives plain-Python objects, not `GstBase` elements: there is no
GObject type system and no `Gst.FlowReturn`. But leaf element code declares
`@GObject.Property(...)` for its tunables and returns `FlowReturn.OK/ERROR` from
its per-frame method. These shims let that same leaf code load and run unchanged
under the g2g backend (`from backend import GObject, FlowReturn`).

`GObject.Property` becomes a plain descriptor that stores values on the instance
(or delegates to a getter/setter pair); gst-only metadata (nick, blurb, min/max)
is accepted and ignored. `FlowReturn` is a plain enum whose `OK`/`ERROR` the host
treats as success/failure of a frame.
"""

from enum import IntEnum


class FlowReturn(IntEnum):
    """The subset of `Gst.FlowReturn` leaf per-frame methods return."""

    OK = 0
    ERROR = -5
    NOT_LINKED = -1
    EOS = -3


class Property:
    """Descriptor mimicking `GObject.Property`, in both forms leaves use.

    Decorator form (getter + optional setter)::

        @GObject.Property(type=str)
        def model_name(self): ...
        @model_name.setter
        def model_name(self, value): ...

    Attribute form (plain storage)::

        broker = GObject.Property(type=str, default=None, nick="...", blurb="...")

    Values back a per-instance ``_g2gprop_<name>`` attribute. The ``type`` /
    ``default`` / ``nick`` / ``blurb`` / ``minimum`` / ``maximum`` keywords are
    accepted for source compatibility and otherwise ignored (no GObject type
    system here).
    """

    def __init__(self, fget=None, *, type=None, default=None, **_gst_meta):
        self._fget = fget
        self._fset = None
        self._default = default
        self._name = None

    # `GObject.Property(type=str)` returns an instance that is then applied as a
    # decorator to the getter; this captures it.
    def __call__(self, fget):
        self._fget = fget
        return self

    def setter(self, fset):
        self._fset = fset
        return self

    def __set_name__(self, owner, name):
        self._name = name

    def _slot(self):
        return f"_g2gprop_{self._name}"

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        if self._fget is not None:
            return self._fget(obj)
        return getattr(obj, self._slot(), self._default)

    def __set__(self, obj, value):
        if self._fset is not None:
            self._fset(obj, value)
        else:
            setattr(obj, self._slot(), value)


class _GObjectShim:
    """The ``GObject`` namespace leaves import from the backend."""

    Property = Property

    # Some leaves type-hint setters as `prop: GObject.GParamSpec`.
    class GParamSpec:  # noqa: N801
        pass


GObject = _GObjectShim()
