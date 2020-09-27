from timeit import default_timer as timer


def timed(method):
    def time_me(*args, **kw):
        start = timer()
        result = method(*args, **kw)
        end = timer()
        print("{!r}  {:4f} s".format(method.__name__, end - start))
        return result

    return time_me
