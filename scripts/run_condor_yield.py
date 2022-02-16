import sys
import time

from esutil.pbar import PBar
from mattspy import BNLCondorParallel
import joblib


def fun(n):
    time.sleep(120)
    return n


def main():
    n_jobs = int(sys.argv[1])

    with BNLCondorParallel(verbose=100, n_jobs=200) as exc:
        tot = 0
        for res in PBar(
            exc([joblib.delayed(fun)(i) for i in range(n_jobs)]),
            total=n_jobs,
            desc="running jobs",
        ):
            if isinstance(res, Exception):
                print(f"failure: {repr(res)}", flush=True)
            else:
                tot += res

    assert tot == sum(range(n_jobs))


if __name__ == "__main__":
    main()
