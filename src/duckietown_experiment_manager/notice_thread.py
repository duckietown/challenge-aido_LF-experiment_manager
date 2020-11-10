import functools
import time
from contextlib import contextmanager
from threading import Thread
from typing import Callable

from . import logger

P = functools.partial

__all__ = ["notice_thread"]


@contextmanager
def notice_thread(msg: str, interval: float):
    stop = False
    t0 = time.time()
    t = Thread(target=notice_thread_child, args=(msg, interval, lambda: stop))
    t.start()
    try:

        yield

    finally:
        t1 = time.time()
        delta = t1 - t0
        logger.info(f"{msg}: took {delta:.1f} seconds.")
        stop = True
        logger.info("waiting for thread to finish")
        t.join()


def notice_thread_child(msg: str, interval: float, stop_condition: Callable[[], bool]) -> None:
    t0 = time.time()
    while not stop_condition():
        delta = time.time() - t0
        logger.info(msg + f"(running for {int(delta)} seconds)")
        time.sleep(interval)
    # logger.info('notice_thread_child finishes')
