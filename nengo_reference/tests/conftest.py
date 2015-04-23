import pytest

import nengo
import nengo_reference

from nengo.tests.conftest import function_seed
from nengo.tests.conftest import plt
from nengo.tests.conftest import seed
from nengo.tests.conftest import rng
from nengo.tests.conftest import RefSimulator
from nengo.tests.conftest import pytest_addoption
from nengo.tests.conftest import pytest_runtest_setup


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
    return nengo_reference.Simulator
