import multiprocessing
import loky
import logging

LOGGER = logging.getLogger("loky_yield")


def _run_func(rd):
    return rd[0](*rd[1], **rd[2])


class LokyYield():
    """A joblib-like interface for the SLAC LSF system that yeilds results.

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
        self, max_workers=None, env=None,
    ):
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.env = env or {}
        self._exec = loky.get_reusable_executor(
            max_workers=self.max_workers,
            env=self.env,
        )

    def __enter__(self):
        self._futs = []
        self._num_jobs = 0
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __call__(self, jobs):
        jobs = iter(jobs)
        done = False
        while True:
            if self._num_jobs < self.max_workers and not done:
                try:
                    job = next(jobs)
                except StopIteration:
                    done = True

                if not done:
                    self._futs.append(self._exec.submit(_run_func, job))
                    self._num_jobs += 1
            else:
                if len(self._futs) == 0:
                    break

                ind = None
                while ind is None:
                    for i in range(len(self._futs)):
                        if self._futs[i].done():
                            ind = i
                            break

                fut = self._futs.pop(ind)
                self._num_jobs -= 1
                try:
                    res = fut.result()
                except Exception as e:
                    res = e

                yield res
