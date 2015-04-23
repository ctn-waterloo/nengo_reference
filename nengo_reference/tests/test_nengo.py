import fnmatch
import os

import pytest
import nengo
from nengo.utils.testing import find_modules, load_functions

nengo_dir = os.path.dirname(nengo.__file__)
modules = find_modules(nengo_dir, prefix='nengo')
modules = [m for m in modules if m[-1] != 'test_pytest']
tests = load_functions(modules, arg_pattern='^Simulator$')


def xfail(pattern, msg):
    for key in tests:
        if fnmatch.fnmatch(key, pattern):
            tests[key] = pytest.mark.xfail(True, reason=msg)(tests[key])

xfail('test.nengo.spa.tests.test_basalganglia.test_basal_ganglia',
      "Requires merging filters.")
xfail('test.nengo.spa.tests.test_thalamus.test_thalamus',
      "Requires merging filters.")
xfail('test.nengo.spa.tests.test_compare.test_run',
      "Requires merging filters.")
xfail('test.nengo.utils.tests.test_connection.test_target_function',
      "Requires merging filters.")

xfail('test.nengo.spa.tests.test_cortical.test_connect',
      "Below tolerance (should pass)")
xfail('test.nengo.spa.tests.test_memory.test_run',
      "Below tolerance (should pass)")

xfail('test.nengo.tests.test_connection.test_shortfilter',
      "No exception for short filter times.")
xfail('test.nengo.tests.test_cache.test_cache_works',
      "No support for passing a built Model into Simulator")
xfail('test.nengo.tests.test_cache.test_cache_performance',
      "No support for passing a built Model into Simulator")

# probing introduces a one-time-step delay (but doesn't in ref Nengo)
xfail('test.nengo.tests.test_connection.test_neurons_to_node',
      "Extra timestep delay")
xfail('test.nengo.tests.test_ensemble.test_constant_scalar',
      "Extra timestep delay")
xfail('test.nengo.tests.test_ensemble.test_scalar', "Extra timestep delay")
xfail('test.nengo.tests.test_ensemble.test_vector', "Extra timestep delay")
xfail('test.nengo.tests.test_node.test_connected', "Extra timestep delay")
xfail('test.nengo.tests.test_node.test_time', "Extra timestep delay")
xfail('test.nengo.tests.test_node.test_simple', "Extra timestep delay")
xfail('test.nengo.tests.test_node.test_passthrough', "Extra timestep delay")
xfail('test.nengo.tests.test_synapses.test_lowpass', "Extra timestep delay")
xfail('test.nengo.tests.test_synapses.test_decoders', "Extra timestep delay")

xfail('test.nengo.tests.test_node.test_args',
      "Connection with synapse=None has a one-time-step delay")

xfail('test.nengo.tests.test_connection.test_weights',
      "Weight-based solvers not supported.")
xfail('test.nengo.tests.test_neurons.test_reset',
      "Weight-based solvers not supported.")

xfail('test.nengo.tests.test_connection.test_nonexistant_prepost',
      "Raises wrong type of error when connecting outside Network")

xfail('test.nengo.tests.test_learning_rules*', "Learning not implemented")


xfail('test.nengo.tests.test_neurons.test_lif',
      "Can't probe voltage or refractory_time")

xfail('test.nengo.tests.test_synapses.test_alpha', "no Alpha synapses")

xfail('test.nengo.tests.test_probe.test_input_probe',
      "can't probe inputs of Ensembles")

xfail('test.nengo.tests.test_neurons.test_alif_rate',
      "specifying ALIF as neuron_type")

xfail('test.nengo.tests.test_probe.test_simulator_dt',
      "raises error with large dt if neurons can't spike as fast as desired")

xfail('test.nengo.tests.test_connection.test_node_to_ensemble',
      "bug in handling Node->Ensemble with pre-slices")

xfail('test.nengo.tests.test_ensemble.test_noise',
      "doesn't handle Ensemble.noise")

xfail('test.nengo.tests.test_neurons.test_lif_zero_tau_ref',
      "can't set tau_ref=0")

xfail('test.nengo.tests.test_neurons.test_izhikevich', "No Izhikevich neurons")

xfail('test.nengo.tests.test_synapses.test_general',
      "No support for synapses other that Lowpass")

xfail('test.nengo.utils.tests.test_ensemble.test_tuning_curves_direct_mode',
      "eval_points not created for Direct mode")

locals().update(tests)
