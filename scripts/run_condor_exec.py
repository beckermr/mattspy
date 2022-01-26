import sys
import time

from concurrent.futures import as_completed
from esutil.pbar import PBar
from mattspy import BNLCondorExecutor


def fun(n):
    time.sleep(120)
    return n


def main():
    n_jobs = int(sys.argv[1])

    with BNLCondorExecutor("bnl", debug=True) as exec:
        futs = [
            exec.submit(fun, i)
            for i in range(n_jobs)
        ]

        sum = 0
        for fut in PBar(as_completed(futs), total=len(futs), desc="running jobs"):
            try:
                sum += fut.result()
            except Exception as e:
                print(f"failure: {repr(e)}", flush=True)

    assert sum == sum(range(n_jobs))


if __name__ == "__main__":
    main()
