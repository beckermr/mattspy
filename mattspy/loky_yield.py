import multiprocessing
import loky
import logging
import concurrent.futures

from .yield_result import ParallelResult

LOGGER = logging.getLogger("loky_yield")


def _run_func(rd):
    return rd[0](*rd[1], **rd[2])


class LokyParallel:
    """A joblib-like interface for the loky parallel backend.

    Parameters
    ----------
    n_jobs : int, optional
        The maximum number of LSF jobs. Default is cpu count on machine.
    env : dict, optional
        Optional environment variables to set in the loky backend.
    verbose : int, optional
        If greater than zero, print debugging information.
    """

    def __init__(
        self,
        n_jobs=-1,
        env=None,
        verbose=0,
    ):
        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        self.env = env or {}
        self._exec = loky.get_reusable_executor(
            max_workers=self.n_jobs,
            env=self.env,
        )
        self.verbose = verbose

    def __enter__(self):
        if self.verbose > 0:
            print(
                "starting LokyParallel(n_jobs=%s, env=%s, verbose=%s)"
                % (
                    self.n_jobs,
                    self.env,
                    self.verbose,
                ),
                flush=True,
            )

        self._futs = []
        self._num_jobs = 0
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def __call__(self, jobs):
        jobs = iter(jobs)
        done = False
        nsub = 0
        index = 0
        while True:
            if self._num_jobs < self.n_jobs * 2 and not done and nsub < 100:
                try:
                    job = next(jobs)
                except StopIteration:
                    done = True

                if not done:
                    fut = self._exec.submit(_run_func, job)
                    fut.index = index
                    self._futs.append(fut)
                    self._num_jobs += 1
                    nsub += 1
                    index += 1
            else:
                nsub = 0

                if len(self._futs) == 0 and done:
                    break

                tp = concurrent.futures.wait(
                    self._futs,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if len(tp[0]) > 0:
                    fut = tp[0].pop()
                    fut = self._futs.pop(self._futs.index(fut))
                    self._num_jobs -= 1

                    try:
                        res = fut.result()
                    except Exception as e:
                        res = e

                    yield ParallelResult(res, fut.index)
