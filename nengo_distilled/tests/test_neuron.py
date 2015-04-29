import pytest
import numpy as np

import nengo
import nengo_distilled.neuron as neuron


def test_lif_rate():
    N = 6
    pool = neuron.LIFRatePool(n_neurons=N)
    input = np.array([-1, 0, 1, 1.001, 5, 10])
    rate = pool.step(input)

    correct = [0, 0, 0, 7.13, 154.73, 243.47]

    assert np.allclose(rate, correct, atol=0.1)


def test_lif():
    N = 100
    pool_rate = neuron.LIFRatePool(n_neurons=N)
    pool_spikes = neuron.LIFSpikingPool(n_neurons=N)
    input = np.linspace(0, 10, N)
    rate = pool_rate.step(input)

    total = np.zeros(N, dtype=float)
    steps = 1000
    for i in range(steps):
        total += pool_spikes.step(input)
    total /= steps

    assert np.allclose(rate, total, rtol=0.08)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
