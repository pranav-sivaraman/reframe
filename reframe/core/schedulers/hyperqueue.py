# Copyright 2016-2026 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# HyperQueue scheduler backend
#
# Jobs are submitted as HyperQueue tasks. HyperQueue's automatic allocation
# requests the allocations for its workers from the workload manager.
#

import enum
import shutil
from dataclasses import dataclass

from reframe.core.backends import register_scheduler
from reframe.core.schedulers import JobScheduler

if shutil.which("hq"):
    error = None
else:
    error = "hq binary not found in PATH"


class _AllocManager(str, enum.Enum):
    """Workload manager that HyperQueue requests allocations from"""

    SLURM = "slurm"


@dataclass(frozen=True)
class _AllocQueueSpec:
    """An `hq alloc add` allocation queue

    Two jobs share an allocation queue if and only if their specs are equal.
    The field names follow the `hq alloc add` options.
    """

    manager: _AllocManager
    time_limit_s: int
    backlog: int
    max_workers_per_alloc: int
    idle_timeout_s: int
    max_worker_count: int | None
    worker_launch_cmd: str | None

    # Passed to the workload manager's submit command after `--`
    rfm_access: tuple[str, ...]

    def command(self, server_dir):
        """The `hq alloc add` command that creates this queue"""

        cmd = [
            "hq",
            "--server-dir",
            str(server_dir),
            "alloc",
            "add",
            self.manager.value,
            f"--time-limit={self.time_limit_s}s",
            f"--backlog={self.backlog}",
            f"--max-workers-per-alloc={self.max_workers_per_alloc}",
            f"--idle-timeout={self.idle_timeout_s}s",
        ]
        if self.max_worker_count is not None:
            cmd.append(f"--max-worker-count={self.max_worker_count}")

        if self.worker_launch_cmd is not None:
            cmd.append(f"--worker-launch-cmd={self.worker_launch_cmd}")

        return [*cmd, "--", *self.rfm_access]


@register_scheduler("hyperqueue", error=error)
class HyperQueueJobScheduler(JobScheduler):
    def __init__(self):
        self._alloc_manager = _AllocManager(
            self.get_option("hq_alloc_manager")
        )
        self._alloc_time_limit_s = self.get_option("hq_alloc_time_limit")
        self._alloc_backlog = self.get_option("hq_alloc_backlog")
        self._alloc_max_worker_count = self.get_option(
            "hq_alloc_max_worker_count"
        )
        self._alloc_idle_timeout_s = self.get_option("hq_alloc_idle_timeout")
        self._alloc_worker_launch_cmd = self.get_option(
            "hq_alloc_worker_launch_cmd"
        )

    def make_job(self, *args, **kwargs):
        raise NotImplementedError

    def emit_preamble(self, job):
        raise NotImplementedError

    def allnodes(self):
        raise NotImplementedError

    def filternodes(self, job, nodes):
        raise NotImplementedError

    def submit(self, job):
        raise NotImplementedError

    def wait(self, job):
        raise NotImplementedError

    def cancel(self, job):
        raise NotImplementedError

    def finished(self, job):
        raise NotImplementedError

    def poll(self, *jobs):
        raise NotImplementedError
