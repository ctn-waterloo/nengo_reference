import logging

import numpy as np
import nengo

import builder

logger = logging.getLogger(__name__)


class Simulator(object):
    """Run a Nengo simulation using the nengo_brainstorm backend."""
    def __init__(self, model, dt=0.001, seed=None):
        if seed is None:
            seed = model.seed
        self.rng = np.random.RandomState(seed=seed)
        self.builder = builder.Builder(model, dt=dt, rng=self.rng)
        self.model = self.builder.model
        self.data = self.model.data
        self.dt = dt
        self.n_steps = 0

    def run(self, time_in_seconds):
        """Simulate for the given length of time."""
        steps = int(np.round(float(time_in_seconds) / self.dt))
        self.run_steps(steps)

    def run_steps(self, steps):
        """Simulate for the given number of `dt` steps."""
        for i in range(steps):
            self.step()

    def reset(self):
        """Reset the simulator state."""
        self.n_steps = 0
        for d in self.model.data.values():
            del d[:]

    def trange(self, dt=None):
        """Create a range of times matching probe data.

        Parameters
        ----------
        dt : float (optional)
            The sampling period of the probe to create a range for. If empty,
            will use the default probe sampling period.
        """
        dt = self.dt if dt is None else dt
        n_steps = int(self.n_steps * (self.dt / dt))
        return dt * np.arange(1, n_steps + 1)

    def step(self):  # noqa: C901

        # keep track of the current time step
        self.n_steps += 1
        self.model.n_steps = self.n_steps
        t = self.n_steps * self.dt

        # simulate Nodes
        for node in self.model.nodes:
            p = self.model.params[node]
            if p.size_in == 0:
                # no input to this Node
                if callable(p.output):
                    output = p.output(t)
                    if output is None and p.size_out > 0:
                        raise ValueError('Node %s returned None' % node)
                    self.model.outputs[node][:] = output
                else:
                    self.model.outputs[node][:] = p.output
            else:
                input = np.sum(self.model.input_filters[node].values(), axis=0)
                for tau, f in self.model.input_filters[node].items():
                    if tau is None or tau == 0:
                        decay = 0
                    else:
                        decay = np.exp(-self.dt/tau)
                    f *= decay
                input.flags.writeable = False
                output = p.output(t, input)
                if p.size_out > 0:
                    if output is None:
                        raise ValueError('Node %s returned None' % node)
                    self.model.outputs[node][:] = output
                else:
                    assert output is None

        # simulate Ensembles
        for ens in self.model.ensembles:
            p = self.model.params[ens]

            # get the input from normal ensemble inputs
            input = np.sum(self.model.input_filters[ens].values(), axis=0)
            if input.shape == ():
                input = np.zeros(ens.size_in)

            # update the synaptic filters
            for tau, f in self.model.input_filters[ens].items():
                if tau is None or tau == 0:
                    decay = 0
                else:
                    decay = np.exp(-self.dt/tau)
                f *= decay

            pool = self.model.pools[ens]

            if pool is None:  # direct mode
                self.model.outputs[ens][:] = input
                continue

            # apply encoder
            encoders = p.scaled_encoders
            input = np.dot(input, encoders.T)

            # get the input that bypasses encoders
            input += np.sum(self.model.input_filters[p.neurons].values(),
                            axis=0) * p.gain / p.radius
            for tau, f in self.model.input_filters[p.neurons].items():
                if tau is None or tau == 0:
                    decay = 0
                else:
                    decay = np.exp(-self.dt/tau)
                f *= decay

            # feed result into neurons
            self.model.outputs[ens][:] = pool.step(input + p.bias)

        # simulate Connections
        for c in self.model.connections:
            pre = c.pre_obj
            post = c.post_obj

            if isinstance(pre, nengo.ensemble.Neurons):
                value = self.model.outputs[pre.ensemble]
            else:
                value = self.model.outputs[pre]
                if isinstance(pre, nengo.Ensemble):
                    if isinstance(pre.neuron_type, nengo.Direct):
                        value = value[c.pre_slice]
                        if c.function is not None:
                            temp = np.zeros((c.size_mid, 1))
                            temp[:] = np.asarray(c.function(value))
                            value = temp
                            if len(value.shape) == 0:
                                value.shape = 1,
                            if len(value.shape) == 2:
                                assert value.shape[1] == 1
                                value.shape = value.shape[0],
                            assert len(value.shape) == 1
                    else:
                        decoder = self.model.params[c].decoders
                        value = np.dot(value, decoder)
                elif isinstance(pre, nengo.Node):
                    if c.function is not None:
                        value = np.asarray(c.function(value))
                        if len(value.shape) == 0:
                            value.shape = 1,
                        assert len(value.shape) == 1

            transform = self.model.params[c].transform
            value = np.dot(value, transform.T)

            tau = c.synapse
            if tau is not None:
                tau = tau.tau
                if tau > 0:
                    value *= 1 - np.exp(-self.dt/tau)

            self.model.input_filters[post][tau] += value
