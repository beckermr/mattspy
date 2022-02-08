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

# TODO
STATUS_DICT = {}

ALL_LSF_JOBS = {}

JOB_TEMPLATE = """\
#!/bin/bash
#BSUB -J "{jobname}"
#BSUB -n 1
#BSUB -o {logfile}
#BSUB -W {timelimit}:00
#BSUB -R "linux64 && rhel60 && scratch > 2"

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
    status = {}
    res = subprocess.run(
        "bjobs %s" % " ".join(cjobs),
        shell=True,
        capture_output=True,
    )
    if res.returncode == 0:
        for line in res.stdout.decode("utf-8").splitlines():
            line = line.strip().split()
            if line[0] == "JOBID":
                continue
            jobid = line[0].strip()
            jobstate = line[2].strip()
            print(line)
            print(jobid, jobstate)
            status[jobid] = jobstate
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


def _submit_lsf_job(exec, subid, nanny_id, fut, job_data, timelimit):
    cjob = None

    if not fut.cancelled():
        infile = os.path.join(exec.execdir, subid, "input.pkl")
        jobfile = os.path.join(exec.execdir, subid, "run.sh")
        outfile = os.path.join(exec.execdir, subid, "output.pkl")
        logfile = os.path.join(exec.execdir, subid, "log.oe")

        os.makedirs(os.path.join(exec.execdir, subid), exist_ok=True)

        ##############################
        # dump the file
        with open(infile, "wb") as fp:
            cloudpickle.dump(job_data, fp)

        ##############################
        # submit the LSF job
        with open(jobfile, "w") as fp:
            fp.write(
                JOB_TEMPLATE.format(
                    input=infile,
                    output=outfile,
                    logfile=logfile,
                    timelimit=timelimit,
                    jobname="job-%s-%s" % (exec.execid, subid),
                )
            )
        subprocess.run(
            "chmod u+x %s" % jobfile,
            shell=True,
            check=True,
        )

        sub = subprocess.run(
            "bsub %s" % jobfile,
            shell=True,
            check=True,
            capture_output=True,
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


def _attempt_submit(exec, nanny_id, subid, timelimit):
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
            cjob = _submit_lsf_job(
                exec, subid, nanny_id, fut, job_data, timelimit,
            )

            if cjob is None:
                LOGGER.debug("could not submit LSF job for subid %s", subid)
                del exec._nanny_subids[nanny_id][subid]
            else:
                LOGGER.debug("submitted LSF job %s for subid %s", cjob, subid)
                fut.cjob = cjob
                exec._nanny_subids[nanny_id][subid] = (cjob, fut, None)
                submitted = True

    return submitted


def _attempt_result(exec, nanny_id, cjob, subids, status_code, debug):
    didit = False
    subid = None
    for _subid in subids:
        if exec._nanny_subids[nanny_id][_subid][0] == cjob:
            subid = _subid
            break
    # TODO
    if subid is not None and status_code in ["DONE", "EXIT"]:
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
        elif status_code in ["EXIT"]:
            res = RuntimeError(
                "LSF job %s: status %s" % (
                    subid, STATUS_DICT[status_code]
                )
            )
        else:
            res = RuntimeError(
                "LSF job %s: no status or job output found!" % subid)

        if not debug:
            subprocess.run(
                "rm -f %s %s %s %s" % (infile, outfile, jobfile, logfile),
                shell=True,
            )

        fut = exec._nanny_subids[nanny_id][subid][1]
        if isinstance(res, Exception):
            fut.set_exception(res)
        else:
            fut.set_result(res)

        exec._nanny_subids[nanny_id][subid] = (None, None, None)
        with ACTIVE_THREAD_LOCK:
            exec._num_jobs -= 1

        didit = True

    return didit


def _nanny_function(
    exec, nanny_id, poll_delay, debug, timelimit,
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
                        if _attempt_submit(exec, nanny_id, subid, timelimit):
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
        Requested time limit in hours.
    verbose : int, optional
        This is ignored but is here for compatability. Use `debug=True`.
    """
    def __init__(
        self, max_workers=10000,
        verbose=0, debug=False, timelimit=10,
    ):
        self.max_workers = max_workers
        self.execid = uuid.uuid4().hex
        self.execdir = "lsf-exec/%s" % self.execid
        self._exec = None
        self._num_nannies = 10
        self.verbose = verbose
        self.debug = debug
        self.timelimit = timelimit

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
            )

    def submit(self, func, *args, **kwargs):
        subid = uuid.uuid4().hex
        job_data = joblib.delayed(func)(*args, **kwargs)

        fut = Future()
        fut.execid = self.execid
        fut.subid = subid
        self._nanny_subids[self._nanny_ind][subid] = (None, fut, job_data)
        fut.set_running_or_notify_cancel()

        self._nanny_ind += 1
        self._nanny_ind = self._nanny_ind % self._num_nannies

        return fut
