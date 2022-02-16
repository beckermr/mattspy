class ParallelResult(object):
    def __init__(self, result_or_exception, index):
        self._result_or_exception = result_or_exception
        self.index = index

    def result(self):
        if isinstance(self._result_or_exception, Exception):
            raise self._result_or_exception
        else:
            return self._result_or_exception


class ParallelSubmissionError(Exception):
    pass
