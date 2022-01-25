import os
import sys
import joblib
import cloudpickle


def run_pickled_task():
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    logfile = sys.argv[3]

    try:
        with open(input_file, "rb") as fp:
            rd = cloudpickle.load(fp)

        try:
            res = rd[0](*rd[1], **rd[2])
        except Exception as e:
            res = e

        joblib.dump(res, output_file)
    finally:
        if not os.path.exists(output_file):
            joblib.dump(RuntimeError("job failed - see %s" % logfile), output_file)
