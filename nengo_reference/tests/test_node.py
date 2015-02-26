import pytest
import numpy as np

import nengo


def test_node_whitenoise(Simulator):
    """Make sure the whitenoise function works

    This is one of the most complex built-in functions from nengo, so
    seems like a good one to test.
    """
    import nengo.utils.functions

    model = nengo.Network()
    with model:
        stimulus = nengo.Node(nengo.processes.WhiteNoise(1, 10).f(), size_out=2)
        a = nengo.Ensemble(100, 2)
        nengo.Connection(stimulus, a)
    sim = Simulator(model)
    sim.run(1)


def test_node_t_type(Simulator):
    """Explicitly test that the t value passed into Nodes is an ndarray."""
    def function(t):
        assert isinstance(t, float)
        return np.sin(t)

    model = nengo.Network()
    with model:
        stimulus = nengo.Node(function, size_out=1)
        a = nengo.Ensemble(100, 1)
        nengo.Connection(stimulus, a)
    sim = Simulator(model)
    sim.run(1)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
