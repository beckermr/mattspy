import os
import uuid
import subprocess
import cloudpickle
import joblib
import atexit
import threading
import logging
import time

from concurrent.futures import ThreadPoolExecutor, as_completed

from .yield_result import ParallelResult, ParallelSubmissionError

LOGGER = logging.getLogger("condor_yield")

ACTIVE_THREAD_LOCK = threading.RLock()

FS_DELAY = 10

ALL_CONDOR_JOBS = {}

STATUS_DICT = {
    None: "unknown condor failure",
    "1": "Idle",
    "2": "Running",
    "3": "Removed",
    "4": "Completed",
    "5": "Held",
    "6": "Transferring Output",
    "7": "Suspended",
    "9": "Killed",
}

WORKER_INIT = """\
#!/bin/bash

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# the condor system creates a scratch directory for us,
# and cleans up afterward
tmpdir=$_CONDOR_SCRATCH_DIR/tmp_me
mkdir -p $tmpdir
export TMPDIR=$tmpdir

mkdir -p $(dirname $2)
mkdir -p $(dirname $3)
touch $3

mattspy-exec-run-pickled-task $1 $2 $3 &> $3
"""


def _kill_condor_jobs():
    chunksize = 100
    cjobs = []
    for cjob in list(ALL_CONDOR_JOBS):
        cjobs.append(cjob)
        if len(cjobs) == chunksize:
            _cjobs = " ".join(cjobs)
            subprocess.run("condor_rm " + _cjobs, shell=True, capture_output=True)
            subprocess.run(
                "condor_rm -forcex " + _cjobs, shell=True, capture_output=True)
            cjobs = []

    if cjobs:
        _cjobs = " ".join(cjobs)
        subprocess.run("condor_rm " + _cjobs, shell=True, capture_output=True)
        subprocess.run("condor_rm -forcex " + _cjobs, shell=True, capture_output=True)
        cjobs = []


def _get_all_job_statuses_call(cjobs):
    res = subprocess.run(
        "condor_q %s -af:jr JobStatus ExitBySignal" % " ".join(cjobs),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if res.returncode == 0:
        status = {c: None for c in cjobs}
        for line in res.stdout.decode("utf-8").splitlines():
            line = line.strip().split(" ")
            if line[0] in cjobs:
                if line[2].strip() == "true":
                    status[line[0]] = "9"
                else:
                    if len(line[1]) > 0:
                        status[line[0]] = line[1]
                    else:
                        status[line[0]] = None
    else:
        status = {}

    return status


def _get_all_job_statuses(cjobs):
    status = {}
    jobs_to_check = []
    for cjob in cjobs:
        jobs_to_check.append(cjob)
        if len(jobs_to_check) == 100:
            status.update(_get_all_job_statuses_call(jobs_to_check))
            jobs_to_check = []

    if jobs_to_check:
        status.update(_get_all_job_statuses_call(jobs_to_check))

    for cjob in list(status):
        if cjob not in cjobs:
            del status[cjob]

    return status


def _submit_condor_job(
    *, execdir, execid, subid, job_data, mem, extra_condor_submit_lines,
):
    infile = os.path.join(execdir, subid, "input.pkl")
    condorfile = os.path.join(execdir, subid, "condor.sub")
    outfile = os.path.join(execdir, subid, "output.pkl")
    logfile = os.path.join(execdir, subid, "log.oe")

    os.makedirs(os.path.join(execdir, subid), exist_ok=True)

    ##############################
    # dump the file
    with open(infile, "wb") as fp:
        cloudpickle.dump(job_data, fp)

    ##############################
    # submit the condor job
    with open(condorfile, "w") as fp:
        fp.write(
            """\
Universe       = vanilla
Notification   = Never
# this executable must have u+x bits
Executable     = %s
request_memory = %dG
kill_sig       = SIGINT
leave_in_queue = True
max_retries    = 0
getenv         = True
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
preserve_relative_paths = True
transfer_input_files = %s
%s

+job_name = "%s"
transfer_output_files = %s,%s
Arguments = %s %s %s
Queue
""" % (
                os.path.join(execdir, "run.sh"),
                mem,
                infile,
                extra_condor_submit_lines,
                "job-%s-%s" % (execid, subid),
                outfile,
                logfile,
                infile,
                outfile,
                logfile,
            ),
        )

    sub = subprocess.run(
        "condor_submit %s" % condorfile,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if sub.returncode != 0 or sub.stdout is None:
        raise ParallelSubmissionError(
            "Error running 'condor_submit %s' - return code %d - stdout+stderr '%s' " % (
                condorfile,
                sub.returncode,
                sub.stdout.decode("utf-8") if sub.stdout is not None else "",
            )
        )

    cjob = None
    for line in sub.stdout.decode("utf-8").splitlines():
        line = line.strip()
        if "submitted to cluster" in line:
            line = line.split(" ")
            cjob = line[5] + "0"
            break

    if cjob is None:
        raise ParallelSubmissionError(
            "Error running 'condor_submit %s' - no job id - return code %d - stdout+stderr '%s'" % (
                condorfile,
                sub.returncode,
                sub.stdout.decode("utf-8") if sub.stdout is not None else "",
            )
        )
    else:
        ALL_CONDOR_JOBS[cjob] = None

        return cjob


def _attempt_submit(*, job_data, execid, execdir, mem, extra_condor_submit_lines):
    subid = uuid.uuid4().hex

    LOGGER.debug("submitting condor job for subid %s", subid)
    try:
        cjob = _submit_condor_job(
            execdir=execdir,
            execid=execid,
            subid=subid,
            job_data=job_data,
            mem=mem,
            extra_condor_submit_lines=extra_condor_submit_lines,
        )
    except Exception as e:
        if not isinstance(e, ParallelSubmissionError):
            e = ParallelSubmissionError(repr(e))
        LOGGER.error("could not submit condor job for subid %s: %s", subid, repr(e))
        raise e

    LOGGER.debug("submitted condor job %s for subid %s", cjob, subid)

    return cjob, subid


class BNLCondorParallel():
    """A joblib-like interface for the BNL condor queue.

    Parameters
    ----------
    n_jobs : int, optional
        The maximum number of condor jobs. Default is 10000.
    mem : int, optional
        Requested memory in GB. Default is 2.
    verbose : int, optional
        If verbose >= 50, all running data will be preserved. Otherwise it is deleted.
    extra_condor_submit_lines : str, optional
        Extra lines of text to pass to the condor submit script.
    """
    def __init__(
        self, n_jobs=10000, verbose=0, mem=2, extra_condor_submit_lines=None,
    ):
        self.n_jobs = n_jobs
        self.execid = uuid.uuid4().hex
        self.execdir = "condor-yield/%s" % self.execid
        self.verbose = verbose
        self.debug = self.verbose >= 50
        self.mem = mem
        self.extra_condor_submit_lines = extra_condor_submit_lines or ""

        if not self.debug:
            atexit.register(_kill_condor_jobs)
        else:
            atexit.unregister(_kill_condor_jobs)

    def __enter__(self):
        os.makedirs(self.execdir, exist_ok=True)
        if self.verbose > 0:
            print(
                "starting BNLCondorParallel("
                f"n_jobs={self.n_jobs}, "
                f"verbose={self.verbose}, "
                f"mem={self.mem}, "
                f"extra_condor_submit_lines={self.extra_condor_submit_lines}) "
                f"w/ exec dir='{self.execdir}'",
                flush=True,
            )

        self._all_jobs = {}
        self._jobid_to_subid = {}
        self._num_jobs = 0
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.debug:
            subprocess.run(
                f"rm -rf {self.execdir}",
                shell=True,
                capture_output=True,
            )

    def __call__(self, jobs):
        with open(os.path.join(self.execdir, "run.sh"), "w") as fp:
            fp.write(WORKER_INIT)
        sub = subprocess.run(
            "chmod u+x " + os.path.join(self.execdir, "run.sh"),
            shell=True,
            capture_output=True,
        )
        if sub.returncode != 0:
            raise ParallelSubmissionError(
                "Could not make writable run script at '%s'!" % os.path.join(self.execdir, "run.sh")
            )

        index = 0
        jobs = iter(jobs)
        done = False
        suberrs = []
        while True:
            # submit
            with ThreadPoolExecutor(max_workers=10) as exc:
                futs = []
                nsub = 0
                while self._num_jobs < self.n_jobs and not done and nsub < 100:
                    try:
                        job = next(jobs)
                    except StopIteration:
                        done = True

                    if not done:
                        futs.append(exc.submit(
                            _attempt_submit,
                            job_data=job,
                            mem=self.mem,
                            execid=self.execid,
                            execdir=self.execdir,
                            extra_condor_submit_lines=self.extra_condor_submit_lines,
                        ))
                        self._num_jobs += 1
                        nsub += 1

                for fut in as_completed(futs):
                    try:
                        cjob, subid = fut.result()
                    except Exception as e:
                        cjob = None
                        err = e

                    if cjob is not None:
                        self._all_jobs[subid] = (cjob, time.time(), index)
                        self._jobid_to_subid[cjob] = subid
                    else:
                        suberrs.append((index, err))
                        self._num_jobs -= 1

                    index += 1

                del futs

            # raise any errors
            for _index, err in suberrs:
                yield ParallelResult(err, _index)
            suberrs = []

            # collect any results
            cjobs = set(
                [tp[0] for tp in self._all_jobs.values() if tp[0] is not None]
            )

            if len(cjobs) == 0 and done:
                return

            statuses = _get_all_job_statuses(cjobs)

            for cjob, status_code in statuses.items():
                didit, res, _index = self._attempt_result(cjob, status_code)
                if didit:
                    yield ParallelResult(res, _index)

    def _attempt_result(self, cjob, status_code):
        didit = False
        subid = self._jobid_to_subid.get(cjob, None)
        res = None
        index = None

        if subid is not None and status_code in [None, "4", "3", "5", "7", "9"]:
            outfile = os.path.join(self.execdir, subid, "output.pkl")
            infile = os.path.join(self.execdir, subid, "input.pkl")
            condorfile = os.path.join(self.execdir, subid, "condor.sub")
            logfile = os.path.join(self.execdir, subid, "log.oe")

            del ALL_CONDOR_JOBS[cjob]
            if not self.debug:
                subprocess.run(
                    "condor_rm %s; condor_rm -forcex %s" % (cjob, cjob),
                    shell=True,
                    capture_output=True,
                )

            if not os.path.exists(outfile):
                time.sleep(FS_DELAY)

            if not os.path.exists(outfile):
                LOGGER.debug(
                    "output %s does not exist for subid %s, condor job %s",
                    outfile,
                    subid,
                    cjob,
                )

            if os.path.exists(outfile):
                try:
                    res = joblib.load(outfile)
                except Exception as e:
                    res = e
            elif status_code in [None, "3", "5", "7", "9"]:
                res = RuntimeError(
                    "Condor job %s: status '%s' w/ no output" % (
                        subid, STATUS_DICT[status_code]
                    )
                )
            else:
                res = RuntimeError(
                    "Condor job %s: no status or job output found!" % subid)

            if isinstance(res, Exception):
                if not self.debug:
                    subprocess.run(
                        "rm -f %s %s %s" % (infile, outfile, condorfile),
                        shell=True,
                        capture_output=True,
                    )
            else:
                if not self.debug:
                    subprocess.run(
                        "rm -f %s %s %s %s" % (infile, outfile, condorfile, logfile),
                        shell=True,
                        capture_output=True,
                    )
                    subprocess.run(
                        "rmdir " + os.path.join(self.execdir, subid),
                        shell=True,
                        capture_output=True,
                    )

            index = self._all_jobs[subid][2]
            self._all_jobs[subid] = (
                None,
                time.time() - self._all_jobs[subid][1],
                self._all_jobs[subid][2],
            )
            self._num_jobs -= 1

            didit = True

        return didit, res, index
