import sys
import time

from esutil.pbar import PBar
from mattspy import SLACLSFYield
import joblib


def fun(n):
    time.sleep(120)
    return n


def main():
    n_jobs = int(sys.argv[1])

    with SLACLSFYield(debug=True, timelimit=10, max_workers=5) as exc:
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