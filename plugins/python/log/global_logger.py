# GlobalLogger
# Copyright (C) 2024-2026 Collabora Ltd.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from log.logger_factory import LoggerFactory
from log.logger import Logger


class GlobalLogger(Logger):
    """Singleton class for logging using the existing LoggerFactory."""

    _instance = None  # Singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.logger = LoggerFactory.get()
        return cls._instance

    def error(self, message, *args):
        self.logger.error(message, *args)

    def warning(self, message, *args):
        self.logger.warning(message, *args)

    def info(self, message, *args):
        self.logger.info(message, *args)

    def debug(self, message, *args):
        self.logger.debug(message, *args)
