import pytest
import numpy as np

import nengo
import nengo_brainstorm.neuron as neuron
import nengo_brainstorm.neuron_tuning as neuron_tuning


def test_rate_approximator(Simulator, plt):
    N = 10
    pool = neuron.LIFSpikingPool(n_neurons=N)

    J = np.linspace(-10, 50, 1000)
    ra = neuron_tuning.RateApproximator()

    a = ra.approximate_activity(pool, J)

    plt.plot(J, a)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
