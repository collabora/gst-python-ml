# ML Alert
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

from log.global_logger import GlobalLogger
import backend

CAN_REGISTER_ELEMENT = True
try:
    import json
    import os
    import time
    import threading
    import urllib.request

    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    gi.require_version("GstVideo", "1.0")
    from gi.repository import Gst, GstBase  # noqa: E402

    from log.logger_factory import LoggerFactory  # noqa: E402
    from backend import analytics, frameio, GObject, post_error  # noqa: E402

    # Header prefix for alert buffer metadata
    ALERT_META_HEADER = b"GST-ALERT:"

    # Building a Gst object needs Gst.init, which only the gst backend calls.
    if backend.BACKEND == "gst":
        VIDEO_SRC_CAPS = Gst.Caps.from_string("video/x-raw")
        VIDEO_SINK_CAPS = Gst.Caps.from_string("video/x-raw")

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_alert' element will not be available. Error: {e}"
    )


class AlertTransform(GstBase.BaseTransform):
    """
    GStreamer element that triggers alerts based on ML detection rules.

    Reads upstream GstAnalytics od_mtd metadata, evaluates configurable rules
    (class name, score threshold, optional zone/ROI), and triggers alerts via
    webhook HTTP POST, MQTT publish, or buffer metadata attachment.
    """

    __gstmetadata__ = (
        "ML Alert",
        "Transform",
        "Triggers alerts based on ML detection rules via webhook, MQTT, or metadata",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    if backend.BACKEND == "gst":
        src_template = Gst.PadTemplate.new(
            "src",
            Gst.PadDirection.SRC,
            Gst.PadPresence.ALWAYS,
            VIDEO_SRC_CAPS.copy(),
        )

        sink_template = Gst.PadTemplate.new(
            "sink",
            Gst.PadDirection.SINK,
            Gst.PadPresence.ALWAYS,
            VIDEO_SINK_CAPS.copy(),
        )
        __gsttemplates__ = (src_template, sink_template)

    @GObject.Property(type=str, default="")
    def rules(self):
        """JSON string defining alert rules, e.g. [{"class": "person", "min_score": 0.8, "zone": [0,0,320,240]}]"""
        return self._rules

    @rules.setter
    def rules(self, value):
        self._rules = value
        self._parse_rules(value)

    webhook_url = GObject.Property(
        type=str,
        default="",
        nick="Webhook URL",
        blurb="HTTP POST endpoint for alert notifications",
        flags=GObject.ParamFlags.READWRITE,
    )

    webhook_token_environment_variable = GObject.Property(
        type=str,
        default="",
        nick="Webhook Token Environment Variable",
        blurb="Environment variable holding the bearer token to send with the webhook",
        flags=GObject.ParamFlags.READWRITE,
    )

    mqtt_topic = GObject.Property(
        type=str,
        default="",
        nick="MQTT Topic",
        blurb="MQTT topic to publish alert messages to",
        flags=GObject.ParamFlags.READWRITE,
    )

    @GObject.Property(type=str, default="")
    def mqtt_broker(self):
        """MQTT broker address (host:port)"""
        return self._mqtt_broker

    @mqtt_broker.setter
    def mqtt_broker(self, value):
        self._mqtt_broker = value
        self._setup_mqtt()

    cooldown = GObject.Property(
        type=int,
        default=10,
        minimum=0,
        maximum=3600,
        nick="Cooldown",
        blurb="Seconds between repeated alerts for the same rule",
        flags=GObject.ParamFlags.READWRITE,
    )

    draw_alert = GObject.Property(
        type=bool,
        default=True,
        nick="Draw Alert",
        blurb="Overlay alert indicator (red border and text) on frame",
        flags=GObject.ParamFlags.READWRITE,
    )

    def __init__(self):
        super().__init__()
        self.logger = LoggerFactory.get(LoggerFactory.LOGGER_TYPE_GST)
        # not passthrough: appending the alert blob needs a writable buffer
        self.set_in_place(True)
        self._rules = ""
        self._mqtt_broker = ""
        self._parsed_rules = []
        self._last_alert_times = {}
        self._mqtt_client = None
        self.width = 0
        self.height = 0

    def _parse_rules(self, rules_json):
        """Parse alert rules from JSON string."""
        if not rules_json:
            self._parsed_rules = []
            return
        try:
            parsed = json.loads(rules_json)
            if isinstance(parsed, dict):
                parsed = [parsed]
            self._parsed_rules = parsed
            self.logger.info(f"Parsed {len(self._parsed_rules)} alert rule(s)")
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse alert rules JSON: {e}")
            self._parsed_rules = []

    def _setup_mqtt(self):
        """Initialize MQTT client if broker is configured and paho is available."""
        if not self.mqtt_broker:
            return
        try:
            import paho.mqtt.client as mqtt

            parts = self.mqtt_broker.split(":")
            host = parts[0]
            port = int(parts[1]) if len(parts) > 1 else 1883
            self._mqtt_client = mqtt.Client()
            self._mqtt_client.connect(host, port, keepalive=60)
            self._mqtt_client.loop_start()
            self.logger.info(f"MQTT connected to {host}:{port}")
        except ImportError:
            self.logger.warning("paho-mqtt not installed, MQTT alerts disabled")
            self._mqtt_client = None
        except Exception as e:
            self.logger.error(f"Failed to connect MQTT broker: {e}")
            self._mqtt_client = None

    def do_set_caps(self, incaps, outcaps):
        s = incaps.get_structure(0)
        self.width = s.get_value("width") or 0
        self.height = s.get_value("height") or 0
        return True

    def _read_detections(self, buf):
        """Extract detections from upstream object-detection metadata."""
        meta = analytics.get_relation_meta(buf)
        if not meta:
            return []
        return analytics.read_objects(meta)

    def _check_rule(self, rule, detection):
        """Check if a detection matches an alert rule."""
        rule_class = rule.get("class", "")
        if rule_class and rule_class not in detection["label"]:
            return False

        min_score = rule.get("min_score", 0.0)
        if detection["score"] < min_score:
            return False

        zone = rule.get("zone")
        if zone and len(zone) == 4:
            zx1, zy1, zx2, zy2 = zone
            dx1, dy1 = detection["x"], detection["y"]
            dx2, dy2 = dx1 + detection["w"], dy1 + detection["h"]
            # Check if detection center is inside the zone
            cx = (dx1 + dx2) / 2.0
            cy = (dy1 + dy2) / 2.0
            if not (zx1 <= cx <= zx2 and zy1 <= cy <= zy2):
                return False

        return True

    def _is_cooled_down(self, rule_idx):
        """Check if enough time has passed since last alert for this rule."""
        now = time.monotonic()
        last = self._last_alert_times.get(rule_idx, 0)
        if now - last >= self.cooldown:
            self._last_alert_times[rule_idx] = now
            return True
        return False

    def _webhook_headers(self):
        headers = {"Content-Type": "application/json"}
        variable = self.webhook_token_environment_variable
        if not variable:
            return headers
        # a named but unset variable would post unauthenticated and read as a server fault
        token = os.environ.get(variable)
        if not token:
            raise ValueError(
                f"webhook token environment variable {variable} is not set"
            )
        headers["Authorization"] = f"Bearer {token}"
        return headers

    def _send_webhook(self, alert_payload):
        """Send alert via HTTP POST in a background thread."""
        if not self.webhook_url:
            return

        headers = self._webhook_headers()

        def _post():
            try:
                data = json.dumps(alert_payload).encode("utf-8")
                req = urllib.request.Request(
                    self.webhook_url,
                    data=data,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    self.logger.info(f"Webhook response: {resp.status}")
            except Exception as e:
                self.logger.error(f"Webhook POST failed: {e}")

        threading.Thread(target=_post, daemon=True).start()

    def _send_mqtt(self, alert_payload):
        """Publish alert to MQTT topic."""
        if not self._mqtt_client or not self.mqtt_topic:
            return
        try:
            data = json.dumps(alert_payload)
            self._mqtt_client.publish(self.mqtt_topic, data)
            self.logger.info(f"MQTT alert published to {self.mqtt_topic}")
        except Exception as e:
            self.logger.error(f"MQTT publish failed: {e}")

    def _attach_alert_meta(self, buf, alert_payload):
        """Attach GST-ALERT: metadata to the buffer."""
        alert_json = json.dumps(alert_payload).encode("utf-8")
        frameio.append_blob(buf, ALERT_META_HEADER, alert_json)

    def _draw_alert_overlay(self, buf):
        """Draw a red border and ALERT text on the frame."""
        import numpy as np

        if self.width == 0 or self.height == 0:
            return
        success, mapinfo = buf.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE)
        if not success:
            return
        # Interpret as RGBA (4 channels)
        frame = np.ndarray(
            (self.height, self.width, 4),
            buffer=mapinfo.data,
            dtype=np.uint8,
        )
        border = max(2, min(self.width, self.height) // 100)
        red = [255, 0, 0, 255]
        # Draw red border
        frame[:border, :] = red
        frame[-border:, :] = red
        frame[:, :border] = red
        frame[:, -border:] = red
        buf.unmap(mapinfo)

    def do_transform_ip(self, buf):
        try:
            if not self._parsed_rules:
                return Gst.FlowReturn.OK

            detections = self._read_detections(buf)
            if not detections:
                return Gst.FlowReturn.OK

            alerts_fired = []
            for rule_idx, rule in enumerate(self._parsed_rules):
                for det in detections:
                    if self._check_rule(rule, det):
                        if not self._is_cooled_down(rule_idx):
                            continue
                        alert_payload = {
                            "timestamp": time.time(),
                            "rule": rule,
                            "detection": det,
                        }
                        alerts_fired.append(alert_payload)
                        self._send_webhook(alert_payload)
                        self._send_mqtt(alert_payload)
                        # Only one alert per rule per frame
                        break

            if alerts_fired:
                self._attach_alert_meta(buf, alerts_fired)
                if self.draw_alert:
                    self._draw_alert_overlay(buf)
                self.logger.info(f"Fired {len(alerts_fired)} alert(s)")

            return Gst.FlowReturn.OK

        except Exception as exception:
            post_error(self, "alert transform error", exception)
            return Gst.FlowReturn.ERROR

    def do_stop(self):
        if self._mqtt_client:
            try:
                self._mqtt_client.loop_stop()
                self._mqtt_client.disconnect()
            except Exception:
                pass
            self._mqtt_client = None
        return True


if CAN_REGISTER_ELEMENT and backend.BACKEND == "gst":
    __gstelementfactory__ = backend.register_gst_element("pyml_alert", AlertTransform)
elif not CAN_REGISTER_ELEMENT:
    GlobalLogger().warning(
        "The 'pyml_alert' element will not be registered because required modules are missing."
    )
