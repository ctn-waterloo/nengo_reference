import pytest

import nengo_distilled

from nengo.tests.conftest import *
from nengo import Direct, LIF, LIFRate, RectifiedLinear, Sigmoid

nt = [Direct, LIF, LIFRate, RectifiedLinear, Sigmoid]

def pytest_generate_tests(metafunc):
    if "nl" in metafunc.funcargnames:
        metafunc.parametrize("nl", nt)
    if "nl_nodirect" in metafunc.funcargnames:
        metafunc.parametrize(
            "nl_nodirect", [n for n in nt if n is not Direct])


@pytest.fixture(scope="session")
def Simulator(request):
    """the Simulator class being tested.

    Please use this, and not nengo.Simulator directly,
    unless the test is reference simulator specific.
    """
    return nengo_distilled.Simulator
