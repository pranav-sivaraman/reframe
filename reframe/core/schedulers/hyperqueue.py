import atexit
import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import reframe.utility.osext as osext
from reframe.core.backends import register_scheduler
from reframe.core.exceptions import (
    JobError,
    JobSchedulerError,
    SpawnedProcessError,
    SpawnedProcessTimeout,
)
from reframe.core.schedulers import JobScheduler, Job

if shutil.which("hq"):
    error = None
else:
    error = (
        "hq binary not found in PATH; install HyperQueue: "
        "https://it4innovations.github.io/hyperqueue/stable/installation/"
    )


class _HyperQueueJob(Job):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        workdir = Path(self.workdir).resolve()
        stdout = workdir / self.stdout
        stderr = workdir / self.stderr

        self._hq_options = [
            f"--name={self.name}",
            f"--cwd={workdir}",
            f"--stdout={stdout}",
            f"--stderr={stderr}",
        ]


@register_scheduler("hyperqueue", error=error)
class HyperQueueJobScheduler(JobScheduler):
    _SERVER_WAIT_TIMEOUT = 30
    _SHUTDOWN_TIMEOUT = 10
    _WORKER_IDLE_TIMEOUT = 30
    _COMPLETED_STATES = ("FINISHED", "FAILED", "CANCELLED", "ABORTED", "TIMEOUT")

    def __init__(self):
        self._alloc_backlog = self.get_option("hq_alloc_backlog")
        self._alloc_manager = self.get_option("hq_alloc_manager")
        self._alloc_time_limit = self.get_option("hq_alloc_time_limit")
        self._alloc_max_worker_count = self.get_option("hq_alloc_max_worker_count")
        self._alloc_max_nodes = 0

        # The server is started lazily on the first submit, since ReFrame
        # creates the scheduler of every partition at startup
        self._server_dir = None
        self._start_error = None

    def _start_server(self):
        if self._start_error:
            raise self._start_error

        if self._server_dir is not None:
            return

        base = Path.home() / ".reframe" / "hq"
        base.mkdir(parents=True, exist_ok=True)
        self._server_dir = Path(tempfile.mkdtemp(prefix="server-", dir=base))

        # Start the hq server in the background
        with (
            open(self._server_dir / "server.out", "w") as stdout,
            open(self._server_dir / "server.err", "w") as stderr,
        ):
            self._server_proc = osext.run_command_async(
                ["hq", "--server-dir", str(self._server_dir), "server", "start"],
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )

        atexit.register(self._shutdown)

        # Wait for the hq server
        try:
            osext.run_command(
                [
                    "hq",
                    "--server-dir",
                    str(self._server_dir),
                    "server",
                    "wait",
                    "--timeout",
                    f"{self._SERVER_WAIT_TIMEOUT}s",
                ],
                check=True,
            )
        except SpawnedProcessError as err:
            # The server directory is removed at exit, so keep its log in
            # the error and fail all later submits the same way
            log = (self._server_dir / "server.err").read_text()
            self._start_error = JobSchedulerError(
                f"failed to start the HyperQueue server:\n{log}"
            )
            raise self._start_error from err

    def _create_alloc_queue(self, access, num_nodes):
        alloc_args = [
            f"--time-limit={self._alloc_time_limit}",
            f"--max-workers-per-alloc={num_nodes}",
            f"--backlog={self._alloc_backlog}",
            # Release idle allocations quickly instead of after HQ's
            # default of 5 minutes
            f"--idle-timeout={self._WORKER_IDLE_TIMEOUT}s",
        ]
        if self._alloc_max_worker_count is not None:
            alloc_args.append(f"--max-worker-count={self._alloc_max_worker_count}")

        # The partition's access options are passed verbatim to sbatch
        manager_args = [
            # Mimic srun's --external-launcher for HQ's worker step: HQ
            # already passes --overlap, so the step doesn't consume any
            # resources; also disable CPU and memory binding, as slurmd
            # does for external launcher steps.
            #
            # The worker step should really use --external-launcher (Slurm
            # >= 23.11), but HQ hardcodes its srun command line. Exporting
            # SLURM_EXTERNAL_LAUNCHER=1 is not an option either, since it
            # leaks to the tasks and turns their srun calls into external
            # launcher steps too. Unlike --external-launcher, this also
            # doesn't ignore --cpus-per-task/--cpus-per-gpu in the access
            # options, which shrink the worker step to those CPUs.
            #
            # TODO: file an issue with HyperQueue to allow customizing the
            # srun command of the worker step
            "--export=ALL,SLURM_CPU_BIND=none,SLURM_MEM_BIND=none",
            *access,
        ]

        osext.run_command(
            [
                "hq",
                "--server-dir",
                str(self._server_dir),
                "alloc",
                "add",
                self._alloc_manager,
                *alloc_args,
                "--",
                *manager_args,
            ],
            check=True,
        )

    def _shutdown(self):
        try:
            osext.run_command(
                ["hq", "--server-dir", str(self._server_dir), "server", "stop"],
                timeout=self._SHUTDOWN_TIMEOUT,
            )
            self._server_proc.wait(timeout=self._SHUTDOWN_TIMEOUT)
        except (SpawnedProcessTimeout, subprocess.TimeoutExpired):
            self._server_proc.kill()
            self._server_proc.wait()

        osext.rmtree(self._server_dir)

    def make_job(self, *args, **kwargs):
        return _HyperQueueJob(*args, **kwargs)

    def emit_preamble(self, job):
        return []

    def allnodes(self):
        raise NotImplementedError("hyperqueue backend does not support node listing")

    def filternodes(self, job, nodes):
        raise NotImplementedError("hyperqueue backend does not support node filtering")

    def submit(self, job):
        self._start_server()

        # TODO: use job.num_nodes once ReFrame exposes it
        num_nodes = (job.num_tasks or 1) // (job.num_tasks_per_node or 1)
        if num_nodes < 1:
            raise JobError(
                f"num_tasks ({job.num_tasks}) must be at least "
                f"num_tasks_per_node ({job.num_tasks_per_node})"
            )

        if num_nodes > self._alloc_max_nodes:
            self._create_alloc_queue(job.sched_access, num_nodes)
            self._alloc_max_nodes = num_nodes

        options = [*job._hq_options, f"--nodes={num_nodes}"]
        if job.time_limit:
            options.append(f"--time-limit={job.time_limit}s")

        script = Path(job.workdir).resolve() / job.script_filename
        completed = osext.run_command(
            [
                "hq",
                "--server-dir",
                str(self._server_dir),
                "--output-mode",
                "json",
                "submit",
                *options,
                str(script),
            ],
            check=True,
        )
        job._jobid = str(json.loads(completed.stdout)["id"])
        job._submit_time = time.time()

    def wait(self, job):
        # Quickly return in case we have finished already
        if self.finished(job):
            return

        osext.run_command(
            ["hq", "--server-dir", str(self._server_dir), "job", "wait", job.jobid],
            check=True,
        )
        self.poll(job)

    def cancel(self, job):
        osext.run_command(
            ["hq", "--server-dir", str(self._server_dir), "job", "cancel", job.jobid],
            check=True,
        )

    def finished(self, job):
        return job.state in self._COMPLETED_STATES

    def poll(self, *jobs):
        if jobs:
            # Filter out non-jobs
            jobs = [job for job in jobs if job is not None]

        if not jobs:
            return

        completed = osext.run_command(
            [
                "hq",
                "--server-dir",
                str(self._server_dir),
                "--output-mode",
                "json",
                "job",
                "info",
                ",".join(job.jobid for job in jobs),
            ],
            check=True,
        )
        infos = {str(info["info"]["id"]): info for info in json.loads(completed.stdout)}
        for job in jobs:
            # ReFrame submits a single task per job
            task = infos[job.jobid]["tasks"][0]
            job._state = task["state"].upper()
            if job.state == "CANCELED":
                # Use the same spelling as the Slurm and Flux backends
                job._state = "CANCELLED"

            if job.state == "FINISHED":
                job._exitcode = 0
            elif job.state == "FAILED":
                error = task.get("error", "")
                if error == "Time limit reached":
                    # Like Slurm: report TIMEOUT and let sanity decide
                    job._state = "TIMEOUT"

                # Only HQ's default message carries the exit code; a task can
                # replace it via HQ_ERROR_FILENAME, so fall back to a non-zero
                # code to never report a failed job as successful
                match = re.fullmatch(
                    r"Program terminated with exit code (-?\d+)\."
                    r"( Received signal .*)?",
                    error,
                )
                job._exitcode = int(match.group(1)) if match else -1
