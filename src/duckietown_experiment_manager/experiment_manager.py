#!/usr/bin/env python
import argparse
import asyncio
import functools
import os
import time
import traceback

from zuper_nodes import RemoteNodeAborted

import duckietown_challenges as dc
from .code import main

P = functools.partial

__all__ = ["go"]


def wrap(cie: dc.ChallengeInterfaceEvaluator) -> None:
    d = cie.get_tmp_dir()
    r = "challenge-evaluation-output/episodes"
    # noinspection PyUnresolvedReferences
    logdir = os.path.join(cie.root, r)

    # logdir = os.path.join(d, "episodes")

    attempts = os.path.join(d, "attempts")
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(attempts):
        os.makedirs(attempts)
    try:
        asyncio.run(main(cie, logdir, attempts), debug=True)
        cie.set_score("simulation-passed", 1)
    except:
        cie.error(f"weird exception: {traceback.format_exc()}")
        raise
    finally:
        cie.info("saving files")
        cie.set_evaluation_dir("episodes", logdir)

    cie.info("experiment_manager::wrap() terminated gracefully.")


def go(args=None):
    parser = argparse.ArgumentParser()
    parsed = parser.parse_args(args)

    import logging
    from procgraph import logger as procgraph_logger

    procgraph_logger.setLevel(logging.ERROR)

    from aido_analyze import logger as aido_analyze_logger

    aido_analyze_logger.setLevel(logging.INFO)

    with dc.scoring_context() as cie:
        try:
            wrap(cie)
        except RemoteNodeAborted as e:

            msg = (
                "It appears that one of the remote nodes has aborted.\n"
                "I will wait 10 seconds before aborting myself so that its\n"
                "error will be detected by the evaluator rather than mine."
            )
            msg += f"\n\n{traceback.format_exc()}"
            cie.error(msg)
            time.sleep(10)
            raise


if __name__ == "__main__":
    go()
