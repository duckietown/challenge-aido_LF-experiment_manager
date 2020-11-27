import os

from duckietown_experiment_manager import (
    ENV_CHALLENGE_NAME,
    ENV_SUBMISSION_ID,
    ENV_SUBMITTER_NAME,
    get_banner_bottom,
)


def test_banner_1():
    os.environ[ENV_SUBMITTER_NAME] = "Andrea Censi"
    os.environ[ENV_SUBMISSION_ID] = "123"
    os.environ[ENV_CHALLENGE_NAME] = "aido5-LF-sim-validation"
    get_banner_bottom("banner_bottom_test.png")
