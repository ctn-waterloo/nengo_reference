import pytest
import numpy as np

import nengo
import nengo.utils.matplotlib


def test_ensemble_probe(Simulator, plt):

    model = nengo.Network()
    with model:
        input = nengo.Node(lambda t: np.sin(2 * np.pi * t))
        a = nengo.Ensemble(100, 1)
        p = nengo.Probe(a, synapse=0.03)

        nengo.Connection(input, a)

    sim = Simulator(model, seed=1)
    sim.run(1.0)

    plt.plot(sim.trange(), sim.data[p])


def test_ensembleArray_probe(Simulator, plt):

    model = nengo.Network()
    with model:
        input = nengo.Node(lambda t: np.sin(2 * np.pi * t))
        a = nengo.networks.EnsembleArray(100, 1)
        p = nengo.Probe(a.output, synapse=0.03)

        nengo.Connection(input, a.input)

    sim = Simulator(model, seed=1)
    sim.run(1.0)

    plt.plot(sim.trange(), sim.data[p])


def test_neuron_probe(Simulator, plt):

    model = nengo.Network()
    with model:
        input = nengo.Node(lambda t: np.sin(2 * np.pi * t))
        a = nengo.Ensemble(10, 1)
        p = nengo.Probe(a.neurons, synapse=None)

        nengo.Connection(input, a)

    sim = Simulator(model, seed=1)
    sim.run(1.0)

    ax = plt.subplot(1, 1, 1)
    nengo.utils.matplotlib.rasterplot(sim.trange(), sim.data[p], ax=ax)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
