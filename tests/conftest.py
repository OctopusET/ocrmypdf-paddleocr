# SPDX-License-Identifier: MPL-2.0

from pathlib import Path

import pytest

RESOURCES = Path(__file__).parent / 'resources'


@pytest.fixture
def resources():
    return RESOURCES
