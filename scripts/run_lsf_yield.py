import sys
import time

from esutil.pbar import PBar
from mattspy import SLACLSFParallel
import joblib


def fun(n):
    time.sleep(120)
    return n


def main():
    n_jobs = int(sys.argv[1])

    with SLACLSFParallel(verbose=100, timelimit=10, n_jobs=200) as exc:
        tot = 0
        for pr in PBar(
            exc([joblib.delayed(fun)(i) for i in range(n_jobs)]),
            total=n_jobs,
            desc="running jobs",
        ):
            try:
                res = pr.result()
            except Exception as e:
                print(f"failure: {repr(e)}", flush=True)
            else:
                tot += res

    assert tot == sum(range(n_jobs))


if __name__ == "__main__":
    main()
