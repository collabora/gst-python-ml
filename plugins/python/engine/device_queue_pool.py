# DeviceQueuePool
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


class DeviceQueue(ABC):
    def __init__(self, queue_handle):
        """
        Initialize the DeviceQueue with a given queue handle.

        :param queue_handle: An integer representing the queue handle.
        """
        self.queue_handle = queue_handle

    @abstractmethod
    def synchronize(self):
        """
        Abstract method to synchronize the device queue.
        Must be implemented by subclasses.
        """
        pass

    def __repr__(self):
        return f"DeviceQueue(handle={self.queue_handle})"


class DeviceQueuePool:
    def __init__(self):
        """
        Initialize the DeviceQueuePool with an empty dictionary to map IDs to DeviceQueues.
        """
        self.queues = {}

    def add_queue(self, queue_id, device_queue):
        """
        Add a DeviceQueue to the pool by its ID.

        :param queue_id: Unique ID for the DeviceQueue.
        :param device_queue: A DeviceQueue object to add to the pool.
        """
        if queue_id in self.queues:
            self.logger.warning(
                f"DeviceQueue with ID {queue_id} already exists. Not adding again."
            )
            return

        self.queues[queue_id] = device_queue

    def get_queue(self, queue_id):
        """
        Retrieve a DeviceQueue by its ID.

        :param queue_id: Unique ID of the queue in the pool.
        :return: DeviceQueue object if found, None otherwise.
        """
        queue = self.queues.get(queue_id, None)
        if queue is None:
            self.logger.warning(f"No DeviceQueue found for ID {queue_id}.")
        return queue

    def __repr__(self):
        return f"DeviceQueuePool(queues={self.queues})"


class DeviceQueueManager:
    _instance = None
    _device_queue_pools = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeviceQueueManager, cls).__new__(cls)
            cls._instance._device_queue_pools = {}
        return cls._instance

    def add_pool(self, device, queue_pool):
        """
        Adds a DeviceQueuePool for a specific device.

        :param device: Unique identifier for the device (e.g., "cuda:0").
        :param queue_pool: A DeviceQueuePool object to associate with the device.
        """
        if device in self._device_queue_pools:
            return  # Do not add if it already exists

        self._device_queue_pools[device] = queue_pool
        self.logger.info(f"Added DeviceQueuePool for device {device}.")

    def get_pool(self, device):
        """
        Retrieves the DeviceQueuePool associated with the specified device.

        :param device: Unique identifier for the device.
        :return: DeviceQueuePool object if found, None otherwise.
        """
        pool = self._device_queue_pools.get(device, None)
        if pool is None:
            self.logger.warning(f"No DeviceQueuePool found for device {device}.")
        return pool

    def __repr__(self):
        return f"DeviceQueueManager(device_queue_pools={self._device_queue_pools})"
