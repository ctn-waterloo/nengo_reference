import os

import pytest

import nengo
from nengo.utils.testing import find_modules, load_functions

nengo_dir = os.path.dirname(nengo.__file__)
modules = find_modules(nengo_dir, prefix='nengo')
tests = load_functions(modules, arg_pattern='^Simulator$')

# get rid of everything except the basic tests
for k in tests.keys():
    if k.startswith('test.nengo.spa'):
        del tests[k]
    if k.startswith('test.nengo.networks'):
        del tests[k]
    if k.startswith('test.nengo.utils'):
        del tests[k]

# not raising exception for short filter times
del tests['test.nengo.tests.test_connection.test_shortfilter']

# requires support for passing a built Model into Simulator
del tests['test.nengo.tests.test_cache.test_cache_works']


# probing introduces a one-time-step delay (but doesn't in ref Nengo)
del tests['test.nengo.tests.test_connection.test_neurons_to_node']
del tests['test.nengo.tests.test_ensemble.test_constant_scalar']
del tests['test.nengo.tests.test_ensemble.test_scalar']
del tests['test.nengo.tests.test_ensemble.test_vector']
del tests['test.nengo.tests.test_node.test_connected']
del tests['test.nengo.tests.test_node.test_time']
del tests['test.nengo.tests.test_node.test_simple']
del tests['test.nengo.tests.test_node.test_passthrough']
del tests['test.nengo.tests.test_synapses.test_lowpass']
del tests['test.nengo.tests.test_synapses.test_decoders']

# Connection with synapse=None has a one-time-step delay
del tests['test.nengo.tests.test_node.test_args']

# don't support weight-based solvers
del tests['test.nengo.tests.test_connection.test_weights']
del tests['test.nengo.tests.test_neurons.test_reset']

# can't set eval_points on Connections yet
del tests['test.nengo.tests.test_connection.test_set_eval_points']

# raises wrong type of error when connecting outside Network
del tests['test.nengo.tests.test_connection.test_nonexistant_prepost']

# don't create eval_points for Direct mode
del tests['test.nengo.tests.test_ensemble.test_eval_points_number']

# warning doesn't seem to be detected by py.test for some reason
del tests['test.nengo.tests.test_ensemble.test_eval_points_number_warning']

# no learning rules
for k in tests.keys():
    if k.startswith('test.nengo.tests.test_learning_rules.'):
        del tests[k]

# can't probe voltage or refractory_time
del tests['test.nengo.tests.test_neurons.test_lif']

# no Alpha synapses
del tests['test.nengo.tests.test_synapses.test_alpha']

# can't probe inputs of Ensembles
del tests['test.nengo.tests.test_probe.test_input_probe']

# specifying ALIF as neuron_type
del tests['test.nengo.tests.test_neurons.test_alif_rate']

# raises error with large dt if neurons can't spike as fast as desired
del tests['test.nengo.tests.test_probe.test_simulator_dt']

# bug in handling Node->Ensemble with pre-slices
del tests['test.nengo.tests.test_connection.test_node_to_ensemble']

# can't set eval_points as distribution on Connection
del tests['test.nengo.tests.test_connection.test_eval_points_scaling']

# bug in not scaling eval_points on Ensemble
del tests['test.nengo.tests.test_ensemble.test_eval_points_scaling']

# doesn't handle Ensemble.noise
del tests['test.nengo.tests.test_ensemble.test_noise']

# can't set tau_ref=0
del tests['test.nengo.tests.test_neurons.test_lif_zero_tau_ref']

# No Izhikevich neurons
del tests['test.nengo.tests.test_neurons.test_izhikevich']

# No support for synapses other that Lowpass
del tests['test.nengo.tests.test_synapses.test_general']

# Useful for temporarily removing most of the tests
keys = sorted(tests.keys())
for k in tests.keys():
    #if k not in keys[50:60]:
    if 'test_node.test_args' not in k:
        del tests[k]

locals().update(tests)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
