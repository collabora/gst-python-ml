# KafkaSink
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

import json

import backend
from backend import BaseSink, GObject, analytics
from utils.analytics_utils import AnalyticsUtils


class KafkaSink(BaseSink):
    GST_PLUGIN_NAME = "mp_kafkasink"

    __gstmetadata__ = (
        "Kafka Sink",
        "Sink",
        "Extracts metadata and sends it to a Kafka server",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    READS_PIXELS = False

    broker = GObject.Property(
        type=str, default=None, nick="Kafka Broker", blurb="Address of the Kafka broker"
    )

    topic = GObject.Property(
        type=str, default=None, nick="Topic", blurb="Kafka topic to send messages to"
    )

    source_id = GObject.Property(
        type=str,
        default="",
        nick="Source ID",
        blurb="Identifier for the metadata source",
    )

    linger_ms = GObject.Property(
        type=int,
        default=0,
        minimum=0,
        maximum=60000,
        nick="Linger Time",
        blurb="Linger time in milliseconds for Kafka producer",
    )

    batch_size = GObject.Property(
        type=int,
        default=10000,
        minimum=1,
        maximum=1000000,
        nick="Batch Size",
        blurb="Maximum number of messages to batch before sending to Kafka",
    )

    message_timeout_ms = GObject.Property(
        type=int,
        default=30000,
        minimum=0,
        maximum=300000,
        nick="Message Timeout",
        blurb="Timeout in milliseconds for message delivery to Kafka",
    )

    compression_type = GObject.Property(
        type=str,
        default="none",
        nick="Compression Type",
        blurb="Type of compression to use for Kafka messages (e.g., none, gzip, snappy, lz4, zstd)",
    )

    acks = GObject.Property(
        type=int,
        default=1,
        minimum=-1,
        maximum=1,
        nick="Acknowledgments",
        blurb="Number of acknowledgments required (0, 1, or -1 for all)",
    )

    def __init__(self):
        super().__init__()
        self.producer = None
        self.label_parser = AnalyticsUtils()

    def initialize_producer(self):
        if self.broker and self.topic:
            try:
                # Imported lazily so a missing confluent_kafka does not break the
                # element scan; it is only needed once a producer is created.
                from confluent_kafka import Producer

                config = {
                    "bootstrap.servers": self.broker,
                    "linger.ms": int(self.linger_ms),
                    "batch.size": int(self.batch_size),
                    "message.timeout.ms": int(self.message_timeout_ms),
                    "compression.type": self.compression_type,
                    "acks": str(self.acks),
                }
                self.producer = Producer(config)
                self.logger.info("Kafka producer initialized successfully.")
            except Exception as e:
                self.logger.error(f"Failed to initialize Kafka producer: {e}")
        else:
            self.logger.warning(
                "Kafka broker or topic is not set, producer not initialized."
            )

    def render(self, frame, fmt, buffer):
        metadata = self.extract_metadata(buffer)
        if not metadata:
            self.logger.warning("No metadata extracted from buffer to send to Kafka.")
            return
        formatted_metadata = self.format_metadata(metadata)
        if formatted_metadata:
            self.send_to_kafka(formatted_metadata)

    def extract_metadata(self, buffer):
        """Extract object detection metadata from GstBuffer using GstAnalyticsRelationMeta."""
        metadata = []

        meta = analytics.get_relation_meta(buffer)
        if not meta:
            self.logger.warning("No analytics metadata found on buffer.")
            return metadata

        try:
            for obj in analytics.read_objects(meta):
                x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
                _, label = self.label_parser.extract_id_from_label(obj["label"])
                detection = {
                    "label": label,
                    "score": obj["score"],
                    "box": {"x1": x, "y1": y, "x2": x + w, "y2": y + h},
                }
                metadata.append(detection)

        except Exception as e:
            self.logger.error(f"Error while extracting metadata: {e}")

        if not metadata:
            self.logger.warning("No metadata extracted from buffer.")
        else:
            self.logger.info(f"Extracted metadata: {metadata}")

        return metadata

    # data/mp_object_detector.json describes this message
    def format_metadata(self, metadata):
        import datetime

        sent = datetime.datetime.now(datetime.timezone.utc)
        return {
            "source_id": self.source_id,
            "timestamp": sent.isoformat().replace("+00:00", "Z"),
            "detections": metadata,
        }

    def send_to_kafka(self, metadata):
        if self.producer is None:
            self.initialize_producer()
            if self.producer is None:
                self.logger.error("Kafka producer is not initialized.")
                return

        metadata_str = json.dumps(metadata)
        self.logger.info(f"Sending metadata to Kafka: {metadata_str}")

        try:
            self.producer.produce(
                self.topic, metadata_str, callback=self.delivery_report
            )
            self.producer.flush()
        except Exception as e:
            self.logger.error(f"Failed to send message to Kafka: {e}")

    def delivery_report(self, err, msg):
        """Callback to handle delivery reports from Kafka."""
        if err is not None:
            self.logger.error(f"Message delivery failed: {err}")
        else:
            self.logger.info(f"Message delivered to {msg.topic()} [{msg.partition()}]")

    def stop(self):
        if self.producer:
            self.producer.flush()


if backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element(
        KafkaSink.GST_PLUGIN_NAME, KafkaSink
    )
