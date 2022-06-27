import time
import joblib

from ..loky_yield import LokyParallel


def test_loky_yield():
    def fun(i):
        time.sleep(1)
        return i

    n_jobs = 100

    tot = 0
    with LokyParallel(n_jobs=2) as exc:
        tot = 0
        for pr in exc([joblib.delayed(fun)(i) for i in range(n_jobs)]):
            try:
                res = pr.result()
            except Exception as e:
                print(f"failure: {repr(e)}", flush=True)
                raise e
            else:
                tot += res

    assert tot == sum(range(n_jobs))
