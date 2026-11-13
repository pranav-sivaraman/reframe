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
import itertools
import json
import math
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import reframe.utility.osext as osext
from reframe.core.backends import register_scheduler
from reframe.core.exceptions import (JobError, JobSchedulerError,
                                     SpawnedProcessError)
from reframe.core.schedulers import Job, JobScheduler

if shutil.which("hq"):
    error = None
else:
    error = "hq binary not found in PATH"

_SERVER_BASE_DIR = Path.home() / ".reframe" / "hq"
_SERVER_WAIT_TIMEOUT_S = 30
_SERVER_STOP_TIMEOUT_S = 10

# HQ task states that end a task, upper-cased as ReFrame's other backends do;
# CANCELLED and TIMEOUT are set by `_update_state()`, see there
_COMPLETED_STATES = ("FINISHED", "FAILED", "CANCELLED", "ABORTED", "TIMEOUT")


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


class _HyperQueueJob(Job):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Directory of the server this job was submitted to; set by `submit()`
        self._hq_server_dir = None


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
        return _HyperQueueJob(*args, **kwargs)

    def emit_preamble(self, job):
        # Scheduler options go to the allocation, through the queue spec
        return []

    def allnodes(self):
        raise NotImplementedError(
            "the hyperqueue backend does not support node listing"
        )

    def filternodes(self, job, nodes):
        raise NotImplementedError(
            "the hyperqueue backend does not support node filtering"
        )

    def _alloc_queue_spec(self, job):
        num_tasks = job.num_tasks or 1
        num_tasks_per_node = job.num_tasks_per_node or 1
        num_nodes = math.ceil(num_tasks / num_tasks_per_node)
        return _AllocQueueSpec(
            manager=self._alloc_manager,
            time_limit_s=self._alloc_time_limit_s,
            backlog=self._alloc_backlog,
            max_workers_per_alloc=num_nodes,
            idle_timeout_s=self._alloc_idle_timeout_s,
            max_worker_count=self._alloc_max_worker_count,
            worker_launch_cmd=self._alloc_worker_launch_cmd,
            rfm_access=tuple(
                job.sched_access + job.options + job.cli_options
            ),
        )

    def submit(self, job):
        spec = self._alloc_queue_spec(job)
        server = _servers.get(spec)

        workdir = Path(job.workdir).resolve()
        submit_cmd = [
            "hq",
            "--server-dir",
            str(server.server_dir),
            "--output-mode",
            "json",
            "submit",
            f"--name={job.name}",
            f"--cwd={workdir}",
            f"--stdout={workdir / job.stdout}",
            f"--stderr={workdir / job.stderr}",
            # Every job gets whole nodes, as with an `sbatch` allocation
            f"--nodes={spec.max_workers_per_alloc}",
        ]
        if job.time_limit:
            submit_cmd.append(f"--time-limit={int(job.time_limit)}s")

        submit_cmd.append(str(workdir / job.script_filename))
        completed = osext.run_command(submit_cmd, check=True)

        job._hq_server_dir = server.server_dir
        job._jobid = str(json.loads(completed.stdout)["id"])
        job._submit_time = time.time()

    def wait(self, job):
        # Poll instead of `hq job wait`, so that `poll()` can cancel a job
        # that exceeds its `max_pending_time`
        intervals = itertools.cycle([1, 2, 3])
        while not self.finished(job):
            self.poll(job)
            time.sleep(next(intervals))

    def cancel(self, job):
        cancel_cmd = [
            "hq",
            "--server-dir",
            str(job._hq_server_dir),
            "job",
            "cancel",
            job.jobid,
        ]
        osext.run_command(cancel_cmd, check=True)

    def finished(self, job):
        return job.state in _COMPLETED_STATES

    def _update_state(self, job, task):
        """Set the state and exit code of `job` from its HQ task"""

        job._state = task["state"].upper()
        if job.state == "CANCELED":
            # Use the same spelling as the Slurm and Flux backends
            job._state = "CANCELLED"
        elif job.state == "FINISHED":
            job._exitcode = 0
        elif job.state == "FAILED":
            error = task.get("error", "")
            if error == "Time limit reached":
                # HQ has no timeout state; report it like Slurm does
                job._state = "TIMEOUT"

            # HQ reports the exit code only in its default error message; a
            # task may replace that message, so fall back to a non-zero code
            match = re.match(r"Program terminated with exit code (-?\d+)\.",
                             error)
            job._exitcode = int(match.group(1)) if match else -1

    def poll(self, *jobs):
        jobs = [job for job in jobs if job is not None]

        # One query per server for all of its jobs
        jobs_by_server_dir = {}
        for job in jobs:
            jobs_by_server_dir.setdefault(job._hq_server_dir, []).append(job)

        for server_dir, server_jobs in jobs_by_server_dir.items():
            info_cmd = [
                "hq",
                "--server-dir",
                str(server_dir),
                "--output-mode",
                "json",
                "job",
                "info",
                ",".join(job.jobid for job in server_jobs),
            ]
            completed = osext.run_command(info_cmd, check=True)
            infos = {
                str(info["info"]["id"]): info
                for info in json.loads(completed.stdout)
            }
            for job in server_jobs:
                # Every HQ job has a single task
                self._update_state(job, infos[job.jobid]["tasks"][0])

        # Cancel jobs that wait too long; a job waits while HQ requests an
        # allocation for it as well
        for job in jobs:
            if (job.state == "WAITING" and job.max_pending_time and
                    time.time() - job.submit_time >= job.max_pending_time):
                self.cancel(job)
                job._exception = JobError("maximum pending time exceeded",
                                          job.jobid)
