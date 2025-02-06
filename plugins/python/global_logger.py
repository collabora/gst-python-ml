# GlobalLogger
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

from log.logger_factory import LoggerFactory
from log.logger import Logger


class GlobalLogger(Logger):
    """Singleton class for logging using the existing LoggerFactory."""

    _instance = None  # Singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.logger = LoggerFactory.get_logger(
                LoggerFactory.GST
            )  # Use GST by default
        return cls._instance

    def error(self, message, *args):
        self.logger.error(message, *args)

    def warning(self, message, *args):
        self.logger.warning(message, *args)

    def info(self, message, *args):
        self.logger.info(message, *args)

    def debug(self, message, *args):
        self.logger.debug(message, *args)
