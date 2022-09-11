import os
import sys
import joblib
import cloudpickle


def run_pickled_task():
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    logfile = sys.argv[3]
    if len(sys.argv) > 4:
        use_exit_code = True if sys.argv[4] == "1" else False
    else:
        use_exit_code = False
    errored = False

    try:
        with open(input_file, "rb") as fp:
            rd = cloudpickle.load(fp)

        try:
            res = rd[0](*rd[1], **rd[2])
        except Exception as e:
            errored = True
            res = e

        joblib.dump(res, output_file)
    finally:
        if not os.path.exists(output_file):
            joblib.dump(RuntimeError("job failed - see %s" % logfile), output_file)
            errored = True

    if errored and use_exit_code:
        sys.exit(-1)
    else:
        sys.exit(0)
