# GstFeatureManager
# Copyright (C) 2024-2025 Collabora Ltd.
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


import os
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

DEBUG_MODE = os.getenv("GST_FEATURE_DEBUG", "0") == "1"


class GstFeatureManager:
    """Singleton class to manage and cache GStreamer capabilities dynamically."""

    _instance = None

    # Only checking for GstAnalytics for now
    STANDARD_CAPABILITIES = ["GstAnalytics"]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GstFeatureManager, cls).__new__(cls)
            cls._instance._checked_features = {}
            cls._instance._imported_objects = {}
            cls._instance._checked_methods = {}
        return cls._instance

    def debug_log(self, message):
        """Logs debug messages if debugging is enabled."""
        if DEBUG_MODE:
            print(f"[{self.__class__.__name__}] {message}")

    def is_available(self, feature_name):
        """Check if GstAnalytics is available."""
        if feature_name in self._checked_features:
            return self._checked_features[feature_name]

        try:
            if feature_name == "GstAnalytics":
                gi.require_version("GstAnalytics", "1.0")
                from gi.repository import GstAnalytics  # noqa: F401

                self._checked_features[feature_name] = True
            else:
                self._checked_features[feature_name] = False
        except (ImportError, ValueError):
            self._checked_features[feature_name] = False

        return self._checked_features[feature_name]

    def import_feature(self, feature_name):
        """Dynamically imports GstAnalytics if available and returns it."""
        if feature_name in self._imported_objects:
            return self._imported_objects[feature_name]

        if self.is_available(feature_name):
            try:
                if feature_name == "GstAnalytics":
                    from gi.repository import GstAnalytics

                    self._imported_objects[feature_name] = GstAnalytics
                    return GstAnalytics
            except ImportError:
                return None

        return None

    def is_method_available(self, module_name, method_name):
        """Check if a specific method exists in a module."""
        key = f"{module_name}.{method_name}"
        if key in self._checked_methods:
            return self._checked_methods[key]

        mod = self.import_feature(module_name)
        if not mod:
            self._checked_methods[key] = False
            return False

        available = hasattr(mod, method_name)
        self._checked_methods[key] = available
        return available

    def safe_add_analytics_meta(self, buf):
        """Safely calls GstAnalytics.buffer_add_analytics_relation_meta if available."""
        GstAnalytics = self.import_feature("GstAnalytics")

        if GstAnalytics and self.is_method_available(
            "GstAnalytics", "buffer_add_analytics_relation_meta"
        ):
            return GstAnalytics.buffer_add_analytics_relation_meta(buf)

        self.debug_log(
            "GstAnalytics or buffer_add_analytics_relation_meta is not available. Skipping metadata."
        )
        return None

    def print_capabilities(self):
        """Prints whether GstAnalytics and its method are available."""
        status = (
            "✔ Available" if self.is_available("GstAnalytics") else "✖ Not Available"
        )
        method_status = (
            "✔ Available"
            if self.is_method_available(
                "GstAnalytics", "buffer_add_analytics_relation_meta"
            )
            else "✖ Not Available"
        )

        print(f"GstAnalytics: {status}")
        print(f"  ├── buffer_add_analytics_relation_meta: {method_status}")
