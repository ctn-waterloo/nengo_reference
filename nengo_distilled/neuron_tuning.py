"""Helper functions for working with neurons."""
import warnings

import numpy as np

try:
    import scipy.interpolate
except ImportError:
    warnings.warn('Could not import "scipy.interpolate"')
    HAS_SCIPY = False


def compute_neuron_response(neurons, time=1, initial_time=0.1, steps=100,
                            J_min=0, J_max=10):
    """Return the average response of the neuron for a range of inputs.

    Parameters
    -----------
    neurons : Pool
        the neurons to examine
    time: float
        amount of time to simulate for
    initial_time: float
        amount of time to simulate for and ignore (to deal with transients)
    steps: int
        the number of different J inputs to consider
    J_min: float
        the minimum J input value
    J_max: float
        the maximum J input value
    """
    assert neurons.identical_neurons  # assumes all neurons are identical

    pool = neurons.create_pool(steps)

    input = np.linspace(J_min, J_max, steps)
    for i in range(int(initial_time / pool.dt)):
        pool.step(input)

    rate = compute_firing_rate(pool, input, time=time, initial_time=0.1)

    return input, rate


def compute_firing_rate(neurons, J, time=1, initial_time=0.1):
    """Return the average response of the neuron for the input J.

    Parameters
    -----------
    neurons : Pool
        the neurons to use
    J : vector of input currents
        the input for each neuron (same length as neurons.n_neurons)
    time: float
        amount of time to simulate for
    initial_time: float
        amount of time to simulate for and ignore (to deal with transients)
    """
    steps = int(time / neurons.dt)
    total = np.zeros(neurons.n_neurons, dtype=float)
    for i in range(steps):
        total += neurons.step(J)
    return total / steps


def find_gain_bias(neurons, intercepts, max_rates):
    """Find the gains and biases for the desired interceps and max_rates.

    This takes the neurons, approximates their response function, and then
    uses that approximation to find the gain and bias value that will give
    the requested intercepts and max_rates.

    Parameters
    -----------
    neurons : Pool
        the neurons to use
    intercepts : vector of floats
        the desired x-intercept for each neuron
    max_rates : vector of floats
        the desired maximum firing rate for each neuron
    """
    J_max = 10
    max_rate = np.max(max_rates)
    J, rate = compute_neuron_response(neurons, J_min=0, J_max=J_max)
    while rate[-1] < max_rate and J_max < 100:
        J_max += 10
        J, rate = compute_neuron_response(neurons, J_min=0, J_max=J_max)
    if J_max >= 100:
        raise ValueError('Could not get %s neurons to fire at %1.2fHz' %
                         (neurons, max_rate))
    J_threshold = J[np.where(rate <= 0)[0][-1]]

    gains = np.zeros(neurons.n_neurons)
    biases = np.zeros(neurons.n_neurons)
    for i in range(len(intercepts)):
        index = np.where(rate > max_rates[i])[0]
        if len(index) == 0:
            index = len(rate) - 1
        else:
            index = index[0]
        if rate[index] == rate[index-1]:
            p = 1
        else:
            p = (max_rates[i]-rate[index-1]) / (rate[index] - rate[index-1])
        J_top = p * J[index] + (1-p) * J[index-1]

        gain = (J_threshold - J_top) / (intercepts[i] - 1)
        bias = J_top - gain
        gains[i] = gain
        biases[i] = bias
    return gains, biases


class RateApproximator(object):
    """System for quickly producing approximate activity matrices.

    Uses the generated response curve to directly generate A given J, rather
    than simulating every possible input value J.
    """
    def __init__(self, J_min=0, J_max=40):
        self.J_min = J_min
        self.J_max = J_max
        self.response = {}  # cache of previously used response curves

    def approximate_activity(self, neurons, J):
        """Return the activity of neurons given the input J."""
        if not HAS_SCIPY:
            raise ImportError('Could not import "scipy.interpolate"')
        J = np.clip(J, self.J_min, self.J_max)

        r = self.response.get(neurons.neuron_name, None)
        if r is None:
            input, rate = compute_neuron_response(neurons, J_min=self.J_min,
                                                  J_max=self.J_max)
            r = scipy.interpolate.interp1d(input, rate)
            self.response[neurons.neuron_name] = r
        return r(J)
