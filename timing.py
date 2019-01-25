import time as _time


def curr_time():
    return _time.strftime("%Y-%m-%d %H:%M:%S", _time.gmtime())


def format_epoch_time(epoch_time):
    return _time.strftime('%Y-%m-%d %H:%M:%S', _time.gmtime(epoch_time))
