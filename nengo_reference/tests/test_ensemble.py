import pytest
import numpy as np

import nengo


def test_communication(Simulator, RefSimulator, plt, nl):
    D = 2
    N = 50 * D
    radius = 10
    model = nengo.Network()
    with model:
        input = nengo.Node(lambda t: radius * np.sin(2 * np.pi * t))
        a = nengo.Ensemble(n_neurons=N, dimensions=D, neuron_type=nl(),
                           radius=radius)
        b = nengo.Ensemble(n_neurons=N, dimensions=D, neuron_type=nl(),
                           radius=radius)
        nengo.Connection(input, a[0])
        nengo.Connection(a, b)
        pA = nengo.Probe(a, synapse=0.03)
        pB = nengo.Probe(b, synapse=0.03)

    sim = Simulator(model)
    sim.run(1.0)

    simref = RefSimulator(model)
    simref.run(1.0)

    plt.subplot(1, 2, 1)
    plt.plot(sim.data[pA], label='brainstorm')
    plt.plot(simref.data[pA], label='nengo')
    plt.legend(loc='best')

    plt.subplot(1, 2, 2)
    plt.plot(sim.data[pB], label='brainstorm')
    plt.plot(simref.data[pB], label='nengo')
    plt.legend(loc='best')

    assert np.allclose(sim.data[pA], simref.data[pA], atol=0.1 * radius)
    assert np.allclose(sim.data[pB], simref.data[pB], atol=0.1 * radius)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
