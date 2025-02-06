# KafkaSink
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

from global_logger import GlobalLogger

CAN_REGISTER_ELEMENT = True
try:
    import json
    import os
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstAnalytics", "1.0")
    gi.require_version("GLib", "2.0")
    from gi.repository import Gst, GObject, GLib, GstAnalytics  # noqa: E402

    from utils import runtime_check_gstreamer_version
    from confluent_kafka import Producer
except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_kafkasink' element will not be available. Error {e}"
    )


class KafkaSink(Gst.Element):
    GST_PLUGIN_NAME = "pyml_kafkasink"

    __gstmetadata__ = (
        "Kafka Sink",
        "Sink",
        "Extracts metadata and sends it to a Kafka server",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )
    broker = GObject.Property(
        type=str, default=None, nick="Kafka Broker", blurb="Address of the Kafka broker"
    )

    topic = GObject.Property(
        type=str, default=None, nick="Topic", blurb="Kafka topic to send messages to"
    )

    schema_file = GObject.Property(
        type=str,
        default=None,
        nick="Schema File",
        blurb="Path to the JSON schema file to format metadata",
    )

    source_id = GObject.Property(
        type=str,
        default=None,
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
        runtime_check_gstreamer_version()

        self.sinkpad = Gst.Pad.new_from_template(
            Gst.PadTemplate.new(
                "sink",
                Gst.PadDirection.SINK,
                Gst.PadPresence.ALWAYS,
                Gst.Caps.new_any(),
            )
        )
        self.sinkpad.set_chain_function_full(self.chain)
        self.add_pad(self.sinkpad)
        self.producer = None

    def do_set_property(self, prop: GObject.GParamSpec, value):
        if prop.name == "broker":
            self.broker = value
            self.logger.info(f"Setting Kafka broker to: {self.broker}")
            self.initialize_producer()
        elif prop.name == "topic":
            self.topic = value
            self.logger.info(f"Setting Kafka topic to: {self.topic}")
            self.initialize_producer()
        elif prop.name == "schema-file":
            self.schema_file_path = value
            self.load_schema_from_file(value)
        elif prop.name == "source-id":
            self.source_id = value
            self.logger.info(f"Setting source ID to: {self.source_id}")
        elif prop.name == "linger-ms":
            self.linger_ms = value if value is not None else 0
            self.logger.info(f"Setting Kafka producer linger-ms to: {self.linger_ms}")
        elif prop.name == "batch-size":
            self.batch_size = value if value is not None else 10000
            self.logger.info(f"Setting Kafka producer batch size to: {self.batch_size}")
        elif prop.name == "message-timeout-ms":
            self.message_timeout_ms = value if value is not None else 30000
            self.logger.info(
                f"Setting Kafka producer message timeout to: {self.message_timeout_ms}"
            )
        elif prop.name == "compression-type":
            self.compression_type = value if value is not None else "none"
            self.logger.info(
                f"Setting Kafka producer compression type to: {self.compression_type}"
            )
        elif prop.name == "acks":
            self.acks = value if value is not None else 1
            self.logger.info(f"Setting Kafka producer acks to: {self.acks}")

    def do_get_property(self, prop: GObject.GParamSpec):
        if prop.name == "broker":
            return self.broker
        elif prop.name == "topic":
            return self.topic
        elif prop.name == "schema-file":
            return self.schema_file_path
        elif prop.name == "source-id":
            return self.source_id
        elif prop.name == "linger-ms":
            return self.linger_ms
        elif prop.name == "batch-size":
            return self.batch_size
        elif prop.name == "message-timeout-ms":
            return self.message_timeout_ms
        elif prop.name == "compression-type":
            return self.compression_type
        elif prop.name == "acks":
            return self.acks
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def initialize_producer(self):
        if self.broker and self.topic:
            try:
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

    def load_schema_from_file(self, file_path):
        """Load schema from a JSON file on disk."""
        if os.path.exists(file_path) and os.path.isfile(file_path):
            try:
                with open(file_path, "r") as f:
                    self.schema = json.load(f)
                self.logger.info(f"Loaded schema from {file_path}")
            except (OSError, json.JSONDecodeError) as e:
                self.logger.error(f"Failed to load or parse schema file: {e}")
                self.schema = None
        else:
            self.logger.error(
                f"Schema file {file_path} does not exist or is not a file."
            )

    def chain(self, pad, parent, buffer):
        metadata = self.extract_metadata(buffer)
        if metadata:
            formatted_metadata = self.format_metadata(metadata, buffer.pts)
            if formatted_metadata:
                self.send_to_kafka(formatted_metadata)
        else:
            self.logger.warning("No metadata extracted from buffer to send to Kafka.")
        return Gst.FlowReturn.OK

    def extract_metadata(self, buffer):
        """Extract object detection metadata from GstBuffer using GstAnalyticsRelationMeta."""
        metadata = []

        meta = GstAnalytics.buffer_get_analytics_relation_meta(buffer)
        if not meta:
            self.logger.warning("No GstAnalytics metadata found on buffer.")
            return metadata

        try:
            count = GstAnalytics.relation_get_length(meta)
            for index in range(count):
                ret, od_mtd = meta.get_od_mtd(index)
                if not ret or od_mtd is None:
                    break

                label_quark = od_mtd.get_obj_type()
                label = GLib.quark_to_string(label_quark)
                location = od_mtd.get_location()

                presence, x, y, w, h, loc_conf_lvl = location
                if presence:
                    x1 = x
                    y1 = y
                    x2 = x + w
                    y2 = y + h

                    metadata.append(
                        {
                            "label": label,
                            "confidence": loc_conf_lvl,
                            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        }
                    )
                else:
                    self.logger.warning(
                        "Presence flag in location is False. Skipping this entry."
                    )

        except Exception as e:
            self.logger.error(f"Error while extracting metadata: {e}")

        if not metadata:
            self.logger.warning("No metadata extracted from buffer.")
        else:
            self.logger.info(f"Extracted metadata: {metadata}")

        return metadata

    def format_metadata(self, metadata, timestamp):
        """Format metadata according to the loaded schema or default schema."""
        timestamp_iso = self.convert_timestamp_to_iso(timestamp)

        formatted_metadata = {
            "source_id": self.source_id,
            "timestamp": timestamp_iso,
            "detections": metadata,
        }

        return formatted_metadata

    def convert_timestamp_to_iso(self, timestamp_ns):
        """Convert nanosecond timestamp to ISO 8601 format."""
        import datetime

        timestamp_s = timestamp_ns / Gst.SECOND
        dt = datetime.datetime.utcfromtimestamp(timestamp_s)
        return dt.isoformat() + "Z"

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

    def do_finalize(self):
        if self.producer:
            self.producer.flush()


if CAN_REGISTER_ELEMENT:
    GObject.type_register(KafkaSink)
    __gstelementfactory__ = (KafkaSink.GST_PLUGIN_NAME, Gst.Rank.NONE, KafkaSink)
else:
    GlobalLogger().warning(
        "The 'pyml_kafkasink' element will not be registered because confluent_kafka module is missing."
    )
