import numpy as np

import nengo


class ProbeNodeOutput(object):
    """nengo.Node function to handle collecting Probe data.

    Probes are implemented by adding a Node with this object as a function
    and a Connection to it from the object being probed.  This allows probes
    to use the same logic as Nodes and greatly simplifies situations like
    probing a passthrough Node."""
    def __init__(self, probe, model):
        self.history = []   # the probe data
        model.data.add(probe, self)
        self.probe = probe
        self.model = model

        # sampling period in units of dt
        self.period = (1 if probe.sample_every is None else
                       probe.sample_every / self.model.dt)

    def __call__(self, t, x):
        if self.model.n_steps % self.period < 1:
            self.history.append(x.copy())


class ProbeData(object):
    """The data object that will be accessed as sim.data[<probe>]

    This would be just a dictionary, but we want to guarantee that the
    returned value from the probe is a numpy array.

    If the requested item is not a probe, falls back on using the
    model.params dictionary.
    """
    def __init__(self, model):
        self.data = {}
        self.model = model

    def add(self, probe, data):
        self.data[probe] = data

    def __getitem__(self, key):
        d = self.data.get(key, None)
        if d is not None:
            return np.array(self.data[key].history)
        else:
            return self.model.params[key]
