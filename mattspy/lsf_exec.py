import os
import uuid
import subprocess
import cloudpickle
import joblib
import atexit
import threading
import logging
import time

from concurrent.futures import ThreadPoolExecutor, Future

LOGGER = logging.getLogger("lsf_exec")

ACTIVE_THREAD_LOCK = threading.RLock()

SCHED_DELAY = 120
FS_DELAY = 10

STATUS_DICT = {
    None: "unknown",
    "DONE": "completed",
    "EXIT": "failed+exited",
    "NOT FOUND": "not found",
    "PEND": "pending",
    "SUSP": "suspended",
    "PSUSP": "suspended by owner when pending",
    "USUSP": "suspended by owner when pending",
    "SSUSP": "suspended by systen",
    "RUN": "running",
}

ALL_LSF_JOBS = {}

JOB_TEMPLATE = """\
#!/bin/bash
#BSUB -J "{jobname}"
#BSUB -n {n}
#BSUB -oo ./{logfile}
#BSUB -W {timelimit}
#BSUB -R "linux64 && rhel60 && scratch > 2{mem_str}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p /scratch/$LSB_JOBID
export TMPDIR=/scratch/$LSB_JOBID

mkdir -p $(dirname {output})
mkdir -p $(dirname {logfile})

mattspy-exec-run-pickled-task {input} {output} {logfile}

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


def _get_all_job_statuses_call(cjobs):
    res = subprocess.run(
        "bjobs -a",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    if res.returncode == 0:
        status = {c: None for c in cjobs}
        for line in res.stdout.decode("utf-8").splitlines():
            line = line.strip()
            parts = line.split()
            jobid = None
            if parts[0] == "JOBID":
                continue
            elif "not found" in line:
                jobid = parts[1].replace("<", "").replace(">", "")
                jobstate = "NOT FOUND"
                status[jobid] = jobstate
            else:
                jobid = parts[0].strip()
                jobstate = parts[2].strip()
                status[jobid] = jobstate

            if jobid is not None:
                try:
                    int(jobid)
                    assert jobstate in STATUS_DICT
                except Exception:
                    LOGGER.error("job id and state not parsed: '%s'", line.strip())
    else:
        status = {}

    return status


def _get_all_job_statuses(cjobs):
    status = _get_all_job_statuses_call(cjobs)
    for cjob in list(status):
        if cjob not in cjobs:
            del status[cjob]

    return status


def _fmt_time(timelimit):
    hrs = timelimit // 60
    mins = timelimit - hrs * 60
    return "%02d:%02d" % (hrs, mins)


def _submit_lsf_job(exec, subid, nanny_id, fut, job_data, timelimit, mem):
    cjob = None

    if fut.set_running_or_notify_cancel():
        infile = os.path.join(exec.execdir, subid, "input.pkl")
        jobfile = os.path.join(exec.execdir, subid, "run.sh")
        outfile = os.path.join(exec.execdir, subid, "output.pkl")
        logfile = os.path.join(exec.execdir, subid, "log.oe")

        os.makedirs(os.path.join(exec.execdir, subid), exist_ok=True)

        ##############################
        # dump the file
        with open(infile, "wb") as fp:
            cloudpickle.dump(job_data, fp)

        # compute mem requirement
        if mem > 4:
            mem_str = ' && span[hosts=1]'
            n = 2
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
                    jobname="job-%s-%s" % (exec.execid, subid),
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
        if sub.returncode != 0:
            raise RuntimeError(
                "Error running 'bsub < %s' - return code %d - stdout '%s' - stderr '%s'" % (
                    jobfile,
                    sub.returncode,
                    sub.stdout.decode("utf-8"),
                    sub.stderr.decode("utf-8"),
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
                continue

        assert cjob is not None
        ALL_LSF_JOBS[cjob] = None

    return cjob


def _attempt_submit(exec, nanny_id, subid, timelimit, mem):
    submitted = False
    cjob = exec._nanny_subids[nanny_id][subid][0]
    fut = exec._nanny_subids[nanny_id][subid][1]
    job_data = exec._nanny_subids[nanny_id][subid][2]

    if cjob is None and job_data is not None:
        LOGGER.debug("submitting LSF job for subid %s", subid)
        with ACTIVE_THREAD_LOCK:
            if exec._num_jobs < exec.max_workers:
                exec._num_jobs += 1
                submit_job = True
            else:
                submit_job = False

        if submit_job:
            try:
                cjob = _submit_lsf_job(
                    exec, subid, nanny_id, fut, job_data, timelimit, mem,
                )
                e = "future cancelled"
            except Exception as _e:
                e = repr(_e)
                cjob = None

            if cjob is None:
                LOGGER.error("could not submit LSF job for subid %s: %s", subid, e)
                exec._nanny_subids[nanny_id][subid] = (None, None, None, None)
            else:
                LOGGER.debug("submitted LSF job %s for subid %s", cjob, subid)
                fut.cjob = cjob
                exec._nanny_subids[nanny_id][subid] = (cjob, fut, None, time.time())
                submitted = True

    return submitted


def _attempt_result(exec, nanny_id, cjob, subids, status_code, debug):
    didit = False
    subid = None
    for _subid in subids:
        if exec._nanny_subids[nanny_id][_subid][0] == cjob:
            subid = _subid
            break

    if (
        subid is not None
        and status_code in [None, "NOT FOUND", "DONE", "EXIT"]
        and time.time() - exec._nanny_subids[nanny_id][subid][3] > SCHED_DELAY
    ):
        outfile = os.path.join(exec.execdir, subid, "output.pkl")
        infile = os.path.join(exec.execdir, subid, "input.pkl")
        jobfile = os.path.join(exec.execdir, subid, "run.sh")
        logfile = os.path.join(exec.execdir, subid, "log.oe")

        del ALL_LSF_JOBS[cjob]
        if not debug:
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
        elif status_code in [None, "DONE", "EXIT", "NOT FOUND"]:
            res = RuntimeError(
                "LSF job %s: status '%s' w/ no output" % (
                    subid, STATUS_DICT[status_code]
                )
            )
        else:
            res = RuntimeError(
                "LSF job %s: no status or job output found!" % subid)

        fut = exec._nanny_subids[nanny_id][subid][1]
        if isinstance(res, Exception):
            fut.set_exception(res)
            if not debug:
                subprocess.run(
                    "rm -f %s %s %s" % (infile, outfile, jobfile),
                    shell=True,
                    capture_output=True,
                )
        else:
            fut.set_result(res)
            if not debug:
                subprocess.run(
                    "rm -f %s %s %s %s" % (infile, outfile, jobfile, logfile),
                    shell=True,
                    capture_output=True,
                )

        exec._nanny_subids[nanny_id][subid] = (None, None, None, None)
        with ACTIVE_THREAD_LOCK:
            exec._num_jobs -= 1

        didit = True

    return didit


def _nanny_function(
    exec, nanny_id, poll_delay, debug, timelimit, mem,
):
    LOGGER.info("nanny %d started for exec %s", nanny_id, exec.execid)

    try:
        while True:
            subids = [
                k for k in list(exec._nanny_subids[nanny_id])
                if exec._nanny_subids[nanny_id][k][1] is not None
            ]

            if exec._done and len(subids) == 0:
                break

            if len(subids) > 0:
                n_to_submit = sum(
                    1
                    for subid in subids
                    if (
                        exec._nanny_subids[nanny_id][subid][0] is None
                        and
                        exec._nanny_subids[nanny_id][subid][2] is not None
                    )
                )
                if n_to_submit > 0:
                    n_submitted = 0
                    for subid in subids:
                        if _attempt_submit(exec, nanny_id, subid, timelimit, mem):
                            n_submitted += 1
                        if n_submitted >= 100:
                            break
                elif poll_delay > 0:
                    time.sleep(poll_delay)

                statuses = _get_all_job_statuses([
                    exec._nanny_subids[nanny_id][subid][0]
                    for subid in subids
                    if exec._nanny_subids[nanny_id][subid][0] is not None
                ])
                n_checked = 0
                for cjob, status_code in statuses.items():
                    if _attempt_result(
                        exec, nanny_id, cjob, subids, status_code, debug
                    ):
                        n_checked += 1
                    if n_checked >= 100:
                        break

            elif poll_delay > 0:
                time.sleep(poll_delay)

        subids = [
            k for k in list(exec._nanny_subids[nanny_id])
            if exec._nanny_subids[nanny_id][k][1] is not None
        ]

        LOGGER.info(
            "nanny %d for exec %s is finishing w/ %d subids left",
            nanny_id, exec.execid, subids,
        )
    except Exception as e:
        LOGGER.critical(
            "nanny %d failed! - %s", nanny_id, repr(e)
        )


class SLACLSFExecutor():
    """A concurrent.futures executor for the SLAC LSF system.

    Parameters
    ----------
    max_workers : int, optional
        The maximum number of LSF jobs. Default is 10000.
    debug : bool, optional
        If True, the completed LSF job information is preserved. This can be
        useful to diagnose failures.
    timelimit : int, optional
        Requested time limit in minutes.
    verbose : int, optional
        This is ignored but is here for compatability. Use `debug=True`.
    mem : float, optional
        The required memory in GB. Default is 3.8.
    """
    def __init__(
        self, max_workers=5000,
        verbose=0, debug=False, timelimit=2820, mem=3.8,
    ):
        self.max_workers = max_workers
        self.execid = uuid.uuid4().hex
        self.execdir = "lsf-exec/%s" % self.execid
        self._exec = None
        self._num_nannies = 10
        self.verbose = verbose
        self.debug = debug
        self.timelimit = timelimit
        self.mem = mem

        if not self.debug:
            atexit.register(_kill_lsf_jobs)
        else:
            atexit.unregister(_kill_lsf_jobs)

    def __enter__(self):
        os.makedirs(self.execdir, exist_ok=True)
        if self.debug:
            print(
                "starting LSF executor: "
                "exec dir %s - max workers %s" % (
                    self.execdir,
                    self.max_workers,
                ),
                flush=True,
            )

        self._exec = ThreadPoolExecutor(max_workers=self._num_nannies)
        self._done = False
        self._nanny_subids = [{} for _ in range(self._num_nannies)]
        self._num_jobs = 0
        self._nanny_ind = 0
        self._nanny_futs = [
            self._exec.submit(
                _nanny_function,
                self,
                i,
                max(1, self._num_nannies/10),
                self.debug,
                self.timelimit,
                self.mem,
            )
            for i in range(self._num_nannies)
        ]
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._done = True
        self._exec.shutdown()
        self._exec = None
        if not self.debug:
            subprocess.run(
                f"rm -rf {self.execdir}",
                shell=True,
                capture_output=True,
            )

    def submit(self, func, *args, **kwargs):
        subid = uuid.uuid4().hex
        job_data = joblib.delayed(func)(*args, **kwargs)

        fut = Future()
        fut.execid = self.execid
        fut.subid = subid
        self._nanny_subids[self._nanny_ind][subid] = (None, fut, job_data, None)

        self._nanny_ind += 1
        self._nanny_ind = self._nanny_ind % self._num_nannies

        return fut
