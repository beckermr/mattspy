class ParallelResult(object):
    def __init__(self):
        self._done = False

    def set_result(self, res):
        if not self._done:
            self._res = res
            self._done = True
        else:
            raise RuntimeError("cannot set result if ParallelResult is done!")

    def set_exception(self, exc):
        if not self._done:
            self._exc = exc
            self._done = True
        else:
            raise RuntimeError("cannot set exception if ParallelResult is done!")

    def result(self):
        if not self._done:
            raise RuntimeError("cannot get result/exception if ParallelResult is not done!")
        elif hasattr(self, "_res"):
            return self._res
        elif hasattr(self, "_exc"):
            raise self._exc
