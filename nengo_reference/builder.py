"""Take a nengo model and convert it into a format where it is ready to run."""

import collections

import numpy as np

import nengo
import nengo.utils.numpy as npext
from nengo.dists import Distribution, Uniform
import nengo.utils.builder

import neuron
import neuron_tuning
import probes

# for storing all the generated information about an Ensemble
BuiltEnsemble = collections.namedtuple(
    'BuiltEnsemble', ['eval_points', 'encoders', 'intercepts', 'max_rates',
                      'scaled_encoders', 'gain', 'bias', 'activity',
                      'neurons', 'radius'])

# for storing all the generated information about a Connection
BuiltConnection = collections.namedtuple(
    'BuiltConnection', ['decoders', 'eval_points', 'transform',
                        'solver_info'])

# for storing all the generated information about a Node
BuiltNode = collections.namedtuple(
    'BuiltNode', ['output', 'size_in', 'size_out'])


class Model(object):
    """Storage for all the data to be returned from the Builder.

    Will be accessible after construction as sim.model.
    """
    def __init__(self, dt, rng):
        self.dt = dt       # size of time step
        self.rng = rng     # random number generation
        self.n_steps = 0   # number of time steps so far
        self.params = {}   # all the BuiltEnsembles and BuiltConnections
        self.pools = {}    # mapping from Ensembles to Pools
        self.input_filters = {}    # all the synaptic filters
        self.outputs = {}  # value currently output by a Node or Ensemble
        self.data = probes.ProbeData(self)
        self.decoders = {}  # cache for all computed decoders
        self.nodes = []        # list of all nodes to be run
        self.connections = []  # list of all connections to be run
        self.ensembles = []    # list of all ensembles to be run


class Builder(object):
    """Construct a model and be ready for it to run."""

    def __init__(self, model, dt, rng):
        self.rng = rng   # random number generator
        self.model = Model(dt=dt, rng=rng)
        self.dt = dt

        self.rate_approximator = neuron_tuning.RateApproximator()

        probe_nodes = []
        probe_conns = []
        for probe in model.all_probes:
            node, conn = self.make_probe(probe)
            probe_nodes.append(node)
            probe_conns.append(conn)

        objs = model.all_nodes + model.all_ensembles + probe_nodes
        conns = model.all_connections + probe_conns
        objs, conns = nengo.utils.builder.remove_passthrough_nodes(objs, conns)

        for obj in objs:
            if isinstance(obj, nengo.Node):
                self.make_node(obj)
            elif isinstance(obj, nengo.Ensemble):
                self.make_ensemble(obj)
            else:
                raise Exception('Cannot handle object %s' % obj)
        for conn in conns:
            self.make_connection(conn)

        for conn in self.model.connections:
            self.make_connection_filter(conn)

    def make_pool(self, ens):
        """Figure out what Pool to use for this Ensemble."""
        cls = neuron.determine_best_class(ens.neuron_type)

        return cls(ens.n_neurons, dt=self.dt,
                   tau_ref=ens.neuron_type.tau_ref,
                   tau_rc=ens.neuron_type.tau_rc)

    def make_ensemble(self, ens):
        """Build an Ensemble."""
        self.model.ensembles.append(ens)

        if isinstance(ens.neuron_type, nengo.Direct):
            self.make_direct_ensemble(ens)
            return

        p = self.make_pool(ens)

        if isinstance(ens.encoders, Distribution):
            encoders = ens.encoders.sample(ens.n_neurons, ens.dimensions,
                                           rng=self.rng)
        else:
            encoders = npext.array(ens.encoders, min_dims=2, dtype=np.float64)
            encoders /= npext.norm(encoders, axis=1, keepdims=True)

        intercepts = nengo.builder.ensemble.sample(ens.intercepts,
                                                   ens.n_neurons, rng=self.rng)
        max_rates = nengo.builder.ensemble.sample(ens.max_rates,
                                                  ens.n_neurons, rng=self.rng)

        if ens.gain is not None and ens.bias is not None:
            gain = nengo.builder.ensemble.sample(ens.gain, ens.n_neurons,
                                                 rng=self.rng)
            bias = nengo.builder.ensemble.sample(ens.bias, ens.n_neurons,
                                                 rng=self.rng)
        elif ens.gain is not None or ens.bias is not None:
            raise NotImplementedError("gain or bias set for %s, but not both. "
                                      "Solving for one given the other is not "
                                      "implemented yet." % ens)
        else:
            gain, bias = neuron_tuning.find_gain_bias(p, intercepts, max_rates)

        scaled_encoders = encoders * (gain / ens.radius)[:, np.newaxis]

        self.model.pools[ens] = p

        self.model.decoders[ens] = {}

        if isinstance(ens.eval_points, Distribution):
            n_points = ens.n_eval_points
            if n_points is None:
                n_points = nengo.utils.builder.default_n_eval_points(
                    ens.n_neurons, ens.dimensions)
            eval_points = ens.eval_points.sample(n_points, ens.dimensions,
                                                 self.rng)
            eval_points *= ens.radius
        else:
            if (ens.n_eval_points is not None
                    and ens.eval_points.shape[0] != ens.n_eval_points):
                warnings.warn("Number of eval_points doesn't match "
                              "n_eval_points. Ignoring n_eval_points.")
            eval_points = np.array(ens.eval_points, dtype=np.float64)

        J = gain * np.dot(eval_points, encoders.T / ens.radius) + bias
        activity = self.compute_activity(ens, J)

        self.model.params[ens] = BuiltEnsemble(intercepts=intercepts,
                                               max_rates=max_rates,
                                               gain=gain,
                                               bias=bias,
                                               encoders=encoders,
                                               scaled_encoders=scaled_encoders,
                                               eval_points=eval_points,
                                               activity=activity,
                                               neurons=ens.neurons,
                                               radius=ens.radius
                                               )
        self.model.outputs[ens] = np.zeros(ens.n_neurons, dtype=float)
        self.model.input_filters[ens] = {}
        self.model.input_filters[ens.neurons] = {}

    def make_direct_ensemble(self, ens):
        """Build an Ensemble for a Direct-mode neuron."""
        self.model.pools[ens] = None
        self.model.params[ens] = BuiltEnsemble(intercepts=None,
                                               max_rates=None,
                                               gain=None,
                                               bias=None,
                                               encoders=None,
                                               scaled_encoders=None,
                                               eval_points=None,
                                               activity=None,
                                               neurons=None,
                                               radius=ens.radius
                                               )
        self.model.outputs[ens] = np.zeros(ens.dimensions, dtype=float)
        self.model.input_filters[ens] = {}

    def make_node(self, node):
        """Build a Node."""
        self.model.outputs[node] = np.zeros(node.size_out, dtype=float)
        self.model.input_filters[node] = {}
        self.model.nodes.append(node)
        self.model.params[node] = BuiltNode(output=node.output,
                                            size_in=node.size_in,
                                            size_out=node.size_out)

    def make_connection(self, conn):
        """Build a Connection."""
        self.model.connections.append(conn)
        if isinstance(conn.pre_obj, nengo.Ensemble):
            transform = nengo.utils.builder.full_transform(conn,
                                                           slice_pre=False)
            ens = conn.pre_obj

            if isinstance(ens.neuron_type, nengo.Direct):
                self.model.params[conn] = BuiltConnection(decoders=None,
                                                          eval_points=None,
                                                          transform=transform,
                                                          solver_info=None)
                return

            eval_points = conn.eval_points
            if eval_points is None:
                eval_points = self.model.params[conn.pre_obj].eval_points
            else:
                eval_points = np.array(eval_points, dtype=np.float64)

            if conn.pre_slice == slice(None):
                key = conn.function
            else:
                key = (conn.function, str(conn.pre_slice))
            # TODO: this key should be dependent on eval_points as well

            decoder, solver_info = self.model.decoders[ens].get(key,
                                                                (None, None))
            if decoder is None:
                result = self.compute_decoder(conn, ens, conn.function,
                                              eval_points)
                self.model.decoders[ens][key] = result
                decoder, solver_info = result

            self.model.params[conn] = BuiltConnection(decoders=decoder,
                                                      eval_points=eval_points,
                                                      transform=transform,
                                                      solver_info=solver_info)
        elif isinstance(conn.pre_obj, nengo.Node):
            transform = nengo.utils.builder.full_transform(conn)
            print transform, conn.pre, conn.post
            self.model.params[conn] = BuiltConnection(decoders=None,
                                                      eval_points=None,
                                                      transform=transform,
                                                      solver_info=None)
        elif isinstance(conn.pre_obj, nengo.ensemble.Neurons):
            transform = nengo.utils.builder.full_transform(conn)
            self.model.params[conn] = BuiltConnection(decoders=None,
                                                      eval_points=None,
                                                      transform=transform,
                                                      solver_info=None)
        else:
            raise Exception('Cannot handle connections from %s' % conn.pre_obj)

    def make_connection_filter(self, conn):
        """Ensure that there is a suitable filter available for the connection.

        A standard NEF connection requires a separate filter for each
        different synaptic time constant.
        """
        post = conn.post_obj
        if conn.synapse is None:
            tau = None
        elif isinstance(conn.synapse, nengo.synapses.Lowpass):
            tau = conn.synapse.tau
        else:
            raise Exception('Cannot handle synapse %s' % conn.synapse)

        size = post.size_in
        if tau not in self.model.input_filters[post]:
            self.model.input_filters[post][tau] = np.zeros(size)

    def make_probe(self, probe):
        """Make a Probe by creating a Connection and Node.

        By implementing Proves this way, the rest of the code doesn't need
        to do anything special, as long as it correctly handles
        Connections and Nodes.
        """
        if isinstance(probe.obj, nengo.Node):
            assert probe.attr == 'output'
        elif isinstance(probe.obj, nengo.Ensemble):
            assert probe.attr == 'decoded_output'
        elif isinstance(probe.obj, nengo.ensemble.Neurons):
            assert probe.attr == 'output'
        elif isinstance(probe.obj, nengo.Connection):
            assert probe.attr == 'output'
        else:
            raise Exception('Cannot handle Probe %s' % probe)
        if isinstance(probe.obj, nengo.Connection):
            node = nengo.Node(probes.ProbeNodeOutput(probe, self.model),
                              size_in=probe.size_in, size_out=0,
                              add_to_container=False)
            conn = nengo.Connection(probe.obj.pre, node, synapse=probe.synapse,
                                    function=probe.obj.function,
                                    transform=probe.obj.transform,
                                    add_to_container=False)
        else:
            node = nengo.Node(probes.ProbeNodeOutput(probe, self.model),
                              size_in=probe.size_in, size_out=0,
                              add_to_container=False)
            conn = nengo.Connection(probe.target, node, synapse=probe.synapse,
                                    add_to_container=False)
        return node, conn

    def compute_decoder(self, conn, ens, function, eval_points):
        """Find the decoder for a given Connection."""
        if function is None:
            targets = eval_points[:, conn.pre_slice]
        else:
            targets = np.zeros((len(eval_points), conn.size_mid))
            for i, ep in enumerate(eval_points):
                f = conn.function(ep[conn.pre_slice])
                f = np.asarray(f)

                if len(f.shape) > 1:
                    assert f.shape[1] == 1
                    f.shape = f.shape[0],

                targets[i] = f

        activity = self.model.params[ens].activity

        if conn.solver.weights:
            raise Exception('Not supported yet')
        else:
            decoder, solver_info = conn.solver(activity, targets, rng=self.rng)

        return decoder, solver_info

    def compute_activity(self, ens, J):
        """Compute the activity of an ensemble given the input current J."""
        pool = self.model.pools[ens]

        if pool.identical_neurons:
            flat_J = J.flatten()
            dummy_pool = pool.create_pool(len(flat_J))
            if not pool.has_memory:
                activ = neuron_tuning.compute_firing_rate(dummy_pool, flat_J,
                                                          time=pool.dt,
                                                          initial_time=0)
                activ.shape = J.shape
            elif hasattr(pool, 'training_rate'):
                activ = dummy_pool.training_rate(flat_J)
                activ.shape = J.shape
            else:
                activ = self.rate_approximator.approximate_activity(pool, J)
            return activ

        raise Exception("Can't compute activity for %s" % pool)
