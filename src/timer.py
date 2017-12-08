from time import time

class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time()
        self.lap_start = time()

    def lap(self):
        self.lap_start = time()

    def elapsed(self, evt="Elapsed time"):
        delta = self.report(evt, self.lap_start)
        self.lap()
        return delta

    def total(self, evt="Total time"):
        return self.report(evt, self.start)

    def report(self, evt="timer", reference=None):
        delta = time() - (reference or self.start)
        print("{}: {}".format(evt, delta))
        return delta
        
