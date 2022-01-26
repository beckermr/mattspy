import time

from concurrent.futures import as_completed
from esutil.pbar import PBar
from mattspy import BNLCondorExecutor


def fun():
    time.sleep(120)


def main():
    n_jobs = 2000

    with BNLCondorExecutor("bnl", debug=True) as exec:
        futs = [
            exec.submit(fun)
            for _ in range(n_jobs)
        ]

        for fut in PBar(as_completed(futs), total=len(futs), desc="running jobs"):
            try:
                fut.result()
            except Exception as e:
                print(f"failure: {repr(e)}", flush=True)


if __name__ == "__main__":
    main()
