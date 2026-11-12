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

import atexit
import contextlib
import enum
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import reframe.utility.osext as osext
from reframe.core.backends import register_scheduler
from reframe.core.exceptions import JobSchedulerError, SpawnedProcessError
from reframe.core.schedulers import JobScheduler

if shutil.which("hq"):
    error = None
else:
    error = "hq binary not found in PATH"

_SERVER_BASE_DIR = Path.home() / ".reframe" / "hq"
_SERVER_WAIT_TIMEOUT_S = 30
_SERVER_STOP_TIMEOUT_S = 10


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

    def alloc_add_args(self):
        """The `hq alloc add` arguments that create this queue"""

        cmd = [
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


class _HQServer:
    """An `hq server` process with a single allocation queue"""

    def __init__(self, server_dir):
        self.server_dir = server_dir

    @classmethod
    def start(cls, spec):
        """Start a server with the allocation queue of `spec`

        If the server or its queue cannot be created, the server is stopped
        and a `JobSchedulerError` is raised.
        """

        _SERVER_BASE_DIR.mkdir(parents=True, exist_ok=True)
        server = cls(Path(tempfile.mkdtemp(prefix="server-", dir=_SERVER_BASE_DIR)))

        start_cmd = [
            "hq",
            "--server-dir",
            str(server.server_dir),
            "server",
            "start",
        ]
        with (
            (server.server_dir / "server.out").open("w") as stdout,
            (server.server_dir / "server.err").open("w") as stderr,
        ):
            osext.run_command_async(
                start_cmd,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )

        wait_cmd = [
            "hq",
            "--server-dir",
            str(server.server_dir),
            "server",
            "wait",
            "--timeout",
            f"{_SERVER_WAIT_TIMEOUT_S}s",
        ]
        try:
            osext.run_command(wait_cmd, check=True)
        except SpawnedProcessError as err:
            log = server._log()
            with contextlib.suppress(SpawnedProcessError):
                server.stop()

            raise JobSchedulerError(
                f"failed to start the HyperQueue server:\n{log}"
            ) from err

        alloc_add_cmd = [
            "hq",
            "--server-dir",
            str(server.server_dir),
            *spec.alloc_add_args(),
        ]
        try:
            osext.run_command(alloc_add_cmd, check=True)
        except SpawnedProcessError as err:
            with contextlib.suppress(SpawnedProcessError):
                server.stop()

            raise JobSchedulerError(
                f"failed to create the HyperQueue allocation queue:\n{err}"
            ) from err

        return server

    def stop(self):
        """Stop the server, ignoring errors, and remove its directory"""

        stop_cmd = [
            "hq",
            "--server-dir",
            str(self.server_dir),
            "server",
            "stop",
        ]
        osext.run_command(stop_cmd, timeout=_SERVER_STOP_TIMEOUT_S)

        # TODO: keep the directory if the server failed to start or died,
        # since its logs are the only evidence; this needs a reliable way to
        # tell a clean shutdown from a failure
        osext.rmtree(self.server_dir, ignore_errors=True)

    def _log(self):
        return (self.server_dir / "server.err").read_text()


class _ServerRegistry:
    """Servers by allocation queue spec

    Entries are never removed or replaced, so a spec keeps its server for the
    lifetime of the process. A server that fails to start is not stored, so
    the next job with the same spec tries again.
    """

    def __init__(self):
        self._servers = {}
        atexit.register(self._stop_all)

    def get(self, spec):
        """The server for `spec`, starting it on first use"""

        try:
            return self._servers[spec]
        except KeyError:
            server = _HQServer.start(spec)
            self._servers[spec] = server
            return server

    def _stop_all(self):
        for server in self._servers.values():
            try:
                server.stop()
            except Exception:
                pass


_servers = _ServerRegistry()


@register_scheduler("hyperqueue", error=error)
class HyperQueueJobScheduler(JobScheduler):
    def __init__(self):
        self._alloc_manager = _AllocManager(self.get_option("hq_alloc_manager"))
        self._alloc_time_limit_s = self.get_option("hq_alloc_time_limit")
        self._alloc_backlog = self.get_option("hq_alloc_backlog")
        self._alloc_max_worker_count = self.get_option("hq_alloc_max_worker_count")
        self._alloc_idle_timeout_s = self.get_option("hq_alloc_idle_timeout")
        self._alloc_worker_launch_cmd = self.get_option("hq_alloc_worker_launch_cmd")

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
