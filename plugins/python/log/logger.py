# Logger
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

from abc import ABC, abstractmethod
import logging


class Logger(ABC):
    """Abstract interface for logging."""

    @abstractmethod
    def error(self, message: str, *args):
        pass

    @abstractmethod
    def warning(self, message: str, *args):
        pass

    @abstractmethod
    def info(self, message: str, *args):
        pass

    @abstractmethod
    def debug(self, message: str, *args):
        pass


class PythonLogger(Logger):
    """Logger implementation using Python logging."""

    def __init__(self):
        logging.basicConfig(level=logging.DEBUG)

    def error(self, message, *args):
        logging.error(message % args if args else message)

    def warning(self, message, *args):
        logging.warning(message % args if args else message)

    def info(self, message, *args):
        logging.info(message % args if args else message)

    def debug(self, message, *args):
        logging.debug(message % args if args else message)
