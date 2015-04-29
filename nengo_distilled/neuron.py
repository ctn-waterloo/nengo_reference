import numpy as np

import nengo


class Pool(object):
    def __init__(self, n_neurons, dt, identical_neurons, has_memory,
                 neuron_name):
        """Base class for a group of neurons.
        """
        self.dt = dt  # simulation time step
        self.n_neurons = n_neurons  # number of neurons

        # flag to indicate that all the neurons in the pool are identical
        # (i.e. there is no internal randomness or process variation that
        # affects its performance).  This allows the builder to assume that
        # the response curves are the same across neurons, greatly speeding
        # up the process of finding the activity matrix
        self.identical_neurons = identical_neurons

        # flag to indicate whether the neuron model has any state.  If it
        # does not have state, its output is only dependent on its
        # instantaneous input (e.g. a rate-mode neuron)
        self.has_memory = has_memory

        # a string identifying the neuron type
        self.neuron_name = neuron_name

    def step(self, J):
        """Simulate the neurons given an input J, returning their output."""
        raise NotImplementedError('Pools must provide "step"')

    def create_pool(self, n_neurons):
        """Create a pool with the same properties but different n_neurons."""
        raise NotImplementedError('Pools must provide "create_pool"')


class LIFRatePool(Pool):
    def __init__(self, n_neurons, dt=0.001, tau_rc=0.02, tau_ref=0.002):
        """A floating-point LIF rate-mode neuron."""
        neuron_name = 'LIFRate rc=%g ref=%g' % (tau_rc, tau_ref)
        super(LIFRatePool, self).__init__(n_neurons=n_neurons, dt=dt,
                                          identical_neurons=True,
                                          has_memory=False,
                                          neuron_name=neuron_name)
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref

    def step(self, J):
        """Simulate the neurons given an input J, returning their output."""
        old = np.seterr(divide='ignore', invalid='ignore')
        try:
            r = 1.0 / (self.tau_ref + self.tau_rc * np.log1p(1.0 / (J-1)))
            r[J <= 1] = 0
        finally:
            np.seterr(**old)
        return r

    def create_pool(self, n_neurons):
        """Create a pool with the same properties but different n_neurons."""
        return LIFRatePool(n_neurons, dt=self.dt, tau_rc=self.tau_rc,
                           tau_ref=self.tau_ref)


class LIFSpikingPool(Pool):
    def __init__(self, n_neurons, dt=0.001, tau_rc=0.02, tau_ref=0.002):
        """A floating-point LIF spiking neuron."""
        neuron_name = 'LIFSpike rc=%g ref=%g' % (tau_rc, tau_ref)
        super(LIFSpikingPool, self).__init__(n_neurons=n_neurons, dt=dt,
                                             identical_neurons=True,
                                             has_memory=True,
                                             neuron_name=neuron_name)
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        self.voltage = np.zeros(n_neurons)
        self.refractory_time = np.zeros(n_neurons)

    def training_rate(self, J):
        """Rate-mode approximation to use while training decoders."""
        lif = LIFRatePool(n_neurons=self.n_neurons, dt=self.dt,
                          tau_rc=self.tau_rc, tau_ref=self.tau_ref)
        return lif.step(J)

    def step(self, J):
        """Simulate the neurons given an input J, returning their output."""
        dt = self.dt

        dv = (dt / self.tau_rc) * (J - self.voltage)
        self.voltage += dv

        self.voltage[self.voltage < 0] = 0

        self.refractory_time -= dt

        self.voltage *= (1-self.refractory_time / dt).clip(0, 1)

        spiked = self.voltage > 1

        overshoot = (self.voltage[spiked > 0] - 1) / dv[spiked > 0]
        spiketime = dt * (1 - overshoot)

        self.voltage[spiked > 0] = 0
        self.refractory_time[spiked > 0] = self.tau_ref + spiketime

        return spiked / dt

    def create_pool(self, n_neurons):
        """Create a pool with the same properties but different n_neurons."""
        return LIFSpikingPool(n_neurons, dt=self.dt, tau_rc=self.tau_rc,
                              tau_ref=self.tau_ref)


def determine_best_class(neuron_type):
    """Return the default class to use for different neuron_types."""
    if isinstance(neuron_type, nengo.LIF):
        return LIFSpikingPool
    elif isinstance(neuron_type, nengo.LIFRate):
        return LIFRatePool

    raise Exception('Cannot handle neuron type %s' % neuron_type)
