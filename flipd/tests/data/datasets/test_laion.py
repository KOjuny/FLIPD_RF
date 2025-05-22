import os

import pytest

from data.datasets import LAIONAesthetics


def test_laion_aesthetics():
    is_cicd = os.environ.get("CICD")
    if is_cicd:
        pytest.skip(f"Skipping due to not having data in CI/CD")
    dataset = LAIONAesthetics()
    dataset[0]
