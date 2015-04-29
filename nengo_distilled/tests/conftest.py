import pytest

import nengo
import nengo_distilled

from nengo.tests.conftest import function_seed  # noqa: F401
from nengo.tests.conftest import plt  # noqa: F401
from nengo.tests.conftest import seed  # noqa: F401
from nengo.tests.conftest import rng  # noqa: F401
from nengo.tests.conftest import RefSimulator  # noqa: F401
from nengo.tests.conftest import pytest_addoption  # noqa: F401
from nengo.tests.conftest import pytest_runtest_setup  # noqa: F401


def pytest_generate_tests(metafunc):
    if "nl" in metafunc.funcargnames:
        metafunc.parametrize(
            "nl", [nengo.Direct, nengo.LIF, nengo.LIFRate])
    if "nl_nodirect" in metafunc.funcargnames:
        metafunc.parametrize(
            "nl_nodirect", [nengo.LIF, nengo.LIFRate])


@pytest.fixture(scope="session")
def Simulator(request):
    """the Simulator class being tested.

    Please use this, and not nengo.Simulator directly,
    unless the test is reference simulator specific.
    """
    return nengo_distilled.Simulator
