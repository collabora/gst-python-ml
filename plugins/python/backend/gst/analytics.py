# Analytics store selection
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

import os

import gi

gi.require_version("Gst", "1.0")

from log.logger_factory import LoggerFactory  # noqa: E402

ANALYTICS_STORE_VARIABLE = "PYML_ANALYTICS"
RELATION_META_STORE = "meta"
CHUNK_STORE = "chunk"


def gstanalytics_importable():
    try:
        gi.require_version("GstAnalytics", "1.0")
        from gi.repository import GstAnalytics  # noqa: F401
    except (ImportError, ValueError):
        return False
    return True


def relation_meta_analytics():
    from backend.gst.relation_meta import RelationMetaAnalytics

    return RelationMetaAnalytics()


def chunk_analytics():
    from backend.gst.chunk_analytics import ChunkAnalytics

    return ChunkAnalytics()


def select_analytics():
    logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
    store = os.environ.get(ANALYTICS_STORE_VARIABLE, "").lower()
    if store == RELATION_META_STORE:
        logger.info(
            f"analytics in the relation meta, forced by {ANALYTICS_STORE_VARIABLE}"
        )
        return relation_meta_analytics()
    if store == CHUNK_STORE:
        logger.info(
            f"analytics in a buffer chunk, forced by {ANALYTICS_STORE_VARIABLE}"
        )
        return chunk_analytics()
    if store:
        raise ImportError(
            f"Unknown {ANALYTICS_STORE_VARIABLE}={store!r}, supported stores: "
            f"{RELATION_META_STORE!r}, {CHUNK_STORE!r}"
        )
    if gstanalytics_importable():
        logger.info("analytics in the relation meta, GstAnalytics found")
        return relation_meta_analytics()
    logger.info("analytics in a buffer chunk, GstAnalytics missing")
    return chunk_analytics()
