import os
import uuid
import subprocess
import cloudpickle
import joblib
import atexit
import logging
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

from .yield_result import ParallelResult, ParallelSubmissionError

LOGGER = logging.getLogger("lsf_yield")

SCHED_DELAY = 120
FS_DELAY = 30
POLL_DELAY = 10

STATUS_DICT = {
    None: "none",
    "DONE": "completed",
    "EXIT": "failed+exited",
    "NOT FOUND": "not found",
    "PEND": "pending",
    "SUSP": "suspended",
    "PSUSP": "suspended by owner when pending",
    "USUSP": "suspended by owner when pending",
    "SSUSP": "suspended by systen",
    "RUN": "running",
    "UNKWN": "unknown",
    "ZOMBI": "zombi",
}

ALL_LSF_JOBS = {}

JOB_TEMPLATE = """\
#!/bin/bash
#BSUB -J "{jobname}"
#BSUB -n {n}
#BSUB -oo ./{logfile}
#BSUB -W {timelimit}
#BSUB -R "select[linux64 && rhel60 && scratch > 2]{mem_str}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p /scratch/$LSB_JOBID
export TMPDIR=/scratch/$LSB_JOBID

mkdir -p $(dirname {output})
mkdir -p $(dirname {logfile})

mattspy-exec-run-pickled-task {input} {output} {logfile} 1

rm -rf /scratch/$LSB_JOBID
"""


def _kill_lsf_jobs():
    chunksize = 100
    cjobs = []
    for cjob in list(ALL_LSF_JOBS):
        cjobs.append(cjob)
        if len(cjobs) == chunksize:
            _cjobs = " ".join(cjobs)
            subprocess.run("bkill -s 9 " + _cjobs, shell=True, capture_output=True)
            cjobs = []

    if cjobs:
        _cjobs = " ".join(cjobs)
        subprocess.run("bkill -s 9 " + _cjobs, shell=True, capture_output=True)
        cjobs = []


def _fmt_time(timelimit):
    hrs = timelimit // 60
    mins = timelimit - hrs * 60
    return "%02d:%02d" % (hrs, mins)


def _submit_lsf_job(*, subid, job_data, mem, execid, timelimit, execdir):
    infile = os.path.join(execdir, subid, "input.pkl")
    jobfile = os.path.join(execdir, subid, "run.sh")
    outfile = os.path.join(execdir, subid, "output.pkl")
    logfile = os.path.join(execdir, subid, "log.oe")

    os.makedirs(os.path.join(execdir, subid), exist_ok=True)

    ##############################
    # dump the file
    with open(infile, "wb") as fp:
        cloudpickle.dump(job_data, fp)

    # compute mem requirement
    if mem > 4:
        mem_str = " span[hosts=1]"
        n = math.ceil(mem / 4)
    else:
        n = 1
        mem_str = ""

    ##############################
    # submit the LSF job
    with open(jobfile, "w") as fp:
        fp.write(
            JOB_TEMPLATE.format(
                input=infile,
                output=outfile,
                logfile=logfile,
                timelimit=_fmt_time(timelimit),
                jobname="job-%s-%s" % (execid, subid),
                mem_str=mem_str,
                n=n,
            )
        )
    subprocess.run(
        "chmod u+x %s" % jobfile,
        shell=True,
        check=True,
        capture_output=True,
    )

    sub = subprocess.run(
        "bsub < %s" % jobfile,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if sub.returncode != 0 or sub.stdout is None:
        raise ParallelSubmissionError(
            "Error running 'bsub < %s' - return code %d - stdout+stderr '%s'"
            % (
                jobfile,
                sub.returncode,
                sub.stdout.decode("utf-8") if sub.stdout is not None else "",
            )
        )

    cjob = None
    for line in sub.stdout.decode("utf-8").splitlines():
        line = line.strip()
        line = line.split(" ")
        cjob = line[1].replace("<", "").replace(">", "")
        try:
            int(cjob)
            break
        except Exception:
            cjob = None
            continue

    if cjob is None:
        raise ParallelSubmissionError(
            "Error running 'bsub < %s' - no job id - return code %d - stdout '%s'"
            % (
                jobfile,
                sub.returncode,
                sub.stdout.decode("utf-8") if sub.stdout is not None else "",
            )
        )
    else:
        ALL_LSF_JOBS[cjob] = None
        return cjob


def _attempt_submit(*, job_data, mem, execid, timelimit, execdir):
    subid = uuid.uuid4().hex
    LOGGER.debug("submitting LSF job for subid %s", subid)
    try:
        cjob = _submit_lsf_job(
            subid=subid,
            job_data=job_data,
            mem=mem,
            execid=execid,
            timelimit=timelimit,
            execdir=execdir,
        )
    except Exception as e:
        if not isinstance(e, ParallelSubmissionError):
            e = ParallelSubmissionError(repr(e))
        LOGGER.error("could not submit LSF job for subid %s: %s", subid, e)
        raise e

    LOGGER.debug("submitted LSF job %s for subid %s", cjob, subid)

    return cjob, subid


class SLACLSFParallel:
    """A joblib-like interface for the SLAC LSF system that yeilds results.

    Parameters
    ----------
    n_jobs : int, optional
        The maximum number of LSF jobs. Default of -1 is 3000.
    timelimit : int, optional
        Requested time limit in minutes.
    verbose : int, optional
        If verbose >= 50, all running data will be preserved. Otherwise it is deleted.
    mem : float, optional
        The required memory in GB. Default is 3.8.
    """

    def __init__(
        self,
        n_jobs=-1,
        verbose=0,
        timelimit=2820,
        mem=3.8,
    ):
        self.n_jobs = n_jobs if n_jobs > 0 else 3000
        self.execid = uuid.uuid4().hex
        self.execdir = "lsf-yield/%s" % self.execid
        self.verbose = verbose
        self.debug = True if self.verbose >= 50 else False
        self.timelimit = timelimit
        self.mem = mem

        if not self.debug:
            atexit.register(_kill_lsf_jobs)
        else:
            atexit.unregister(_kill_lsf_jobs)

    def __enter__(self):
        os.makedirs(self.execdir, exist_ok=True)
        if self.verbose > 0:
            print(
                "starting SLACLSFParallel("
                "n_jobs=%s, timelimit=%s, "
                "verbose=%s, mem=%s) w/ exec dir='%s'"
                % (
                    self.n_jobs,
                    self.timelimit,
                    self.verbose,
                    self.mem,
                    self.execdir,
                ),
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
        jobs = iter(jobs)
        done = False
        index = 0
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
                        futs.append(
                            exc.submit(
                                _attempt_submit,
                                job_data=job,
                                mem=self.mem,
                                execid=self.execid,
                                timelimit=self.timelimit,
                                execdir=self.execdir,
                            )
                        )
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
            status_time = time.time()
            cjobs = set([tp[0] for tp in self._all_jobs.values() if tp[0] is not None])

            if len(cjobs) == 0 and done:
                return

            statuses = self._get_all_job_statuses(cjobs)
            status_time = time.time() - status_time
            status_time *= 10

            n_yield = 0
            yield_result_time = time.time()
            for cjob, status_code in statuses.items():
                didit, res, _index = self._attempt_result(cjob, status_code)
                if didit:
                    n_yield += 1
                    yield ParallelResult(res, _index)

                if (
                    not done
                    and self._num_jobs < self.n_jobs
                    and (
                        time.time() - yield_result_time > status_time or n_yield >= 100
                    )
                ):
                    break

    def _attempt_result(self, cjob, status_code):
        didit = False
        res = None
        subid = self._jobid_to_subid.get(cjob, None)
        index = None

        if (
            subid is not None
            and status_code in [None, "NOT FOUND", "DONE", "EXIT", "ZOMBI"]
            and time.time() - self._all_jobs[subid][1] > SCHED_DELAY
        ):
            outfile = os.path.join(self.execdir, subid, "output.pkl")
            infile = os.path.join(self.execdir, subid, "input.pkl")
            jobfile = os.path.join(self.execdir, subid, "run.sh")
            logfile = os.path.join(self.execdir, subid, "log.oe")

            if cjob in ALL_LSF_JOBS:
                del ALL_LSF_JOBS[cjob]
            if not self.debug:
                subprocess.run(
                    "bkill -s 9 %s" % cjob,
                    shell=True,
                    capture_output=True,
                )

            if not os.path.exists(outfile):
                time.sleep(FS_DELAY)

            if not os.path.exists(outfile):
                LOGGER.debug(
                    "output %s does not exist for subid %s, LSF job %s",
                    outfile,
                    subid,
                    cjob,
                )

            if os.path.exists(outfile):
                try:
                    res = joblib.load(outfile)
                except Exception as e:
                    res = e
            elif status_code in ["NOT FOUND", "DONE", "EXIT", "ZOMBI"]:
                res = RuntimeError(
                    "LSF job %s: status '%s' w/ no output"
                    % (subid, STATUS_DICT[status_code])
                )
            else:
                res = RuntimeError("LSF job %s: no status or job output found!" % subid)

            if isinstance(res, Exception):
                if not self.debug:
                    subprocess.run(
                        "rm -f %s %s %s" % (infile, outfile, jobfile),
                        shell=True,
                        capture_output=True,
                    )
            else:
                if not self.debug:
                    subprocess.run(
                        "rm -f %s %s %s %s" % (infile, outfile, jobfile, logfile),
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

    def _get_all_job_statuses(self, cjobs):
        res = subprocess.run(
            "bjobs -aw",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        status = {}

        if res.returncode == 0:
            for line in res.stdout.decode("utf-8").splitlines():
                line = line.strip()
                parts = line.split()
                jobid = None
                if parts[0] == "JOBID":
                    continue
                elif "not responding" in line:
                    # this means the daemon is down so any status info is bad
                    return {}
                elif self.execid not in line:
                    # not our job
                    continue
                elif "not found" in line:
                    jobid = parts[1].replace("<", "").replace(">", "")
                    jobstate = "NOT FOUND"
                    if jobid in cjobs:
                        status[jobid] = jobstate
                else:
                    jobid = parts[0].strip()
                    jobstate = parts[2].strip()
                    if jobid in cjobs:
                        status[jobid] = jobstate

                if jobid is not None:
                    try:
                        int(jobid)
                        assert jobstate in STATUS_DICT
                    except Exception:
                        LOGGER.error("job id and state not parsed: '%s'", line.strip())

        return status
