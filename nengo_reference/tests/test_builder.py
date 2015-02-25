import pytest
import numpy as np

import nengo


def test_spa(Simulator):
    import nengo.spa as spa

    dimensions = 16
    model = spa.SPA()
    with model:
        model.state1 = spa.Buffer(dimensions=dimensions)
        model.state2 = spa.Buffer(dimensions=dimensions)
        model.output = spa.Buffer(dimensions=dimensions)
        actions = spa.Actions('0.5 --> output=state1*state2')
        model.bg = spa.BasalGanglia(actions)
        model.thal = spa.Thalamus(model.bg)

    sim = Simulator(model, seed=1)
    sim.run(0.1)


def test_spa_sequence(Simulator, RefSimulator, plt):
    import nengo.spa as spa

    dimensions = 16
    model = spa.SPA()
    with model:
        model.state = spa.Buffer(dimensions=dimensions)
        model.bg = spa.BasalGanglia(spa.Actions(
            'dot(state, A) --> state=B',
            'dot(state, B) --> state=C',
            'dot(state, C) --> state=D',
            'dot(state, D) --> state=E',
            'dot(state, E) --> state=A',
            ))
        model.thal = spa.Thalamus(model.bg)

        model.input = spa.Input(state=lambda t: 'A' if t < 0.1 else '0')

        p = nengo.Probe(model.thal.actions.output, synapse=0.01)

    sim = Simulator(model, seed=1)
    sim.run(1.0)
    refsim = RefSimulator(model, seed=1)
    refsim.run(1.0)

    plt.subplot(2, 1, 1)
    plt.plot(sim.trange(), sim.data[p])
    plt.subplot(2, 1, 2)
    plt.plot(refsim.trange(), refsim.data[p])


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
