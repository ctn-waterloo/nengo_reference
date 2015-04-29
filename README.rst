===============
nengo_distilled
===============

An example backend implementation for Nengo.

This takes a neural model described
using the Nengo framework,
builds it into an actual neural simulation,
and runs the simulation. For example::

   import numpy as np
   import nengo

   # define the model
   model = nengo.Network()
   with model:
       stim = nengo.Node(np.sin)
       a = nengo.Ensemble(n_neurons=100, dimensions=1)
       b = nengo.Ensemble(n_neurons=100, dimensions=1)
       nengo.Connection(stim, a)
       nengo.Connection(a, b, function=lambda x: x**2, synapse=0.01)

       probe_a = nengo.Probe(a, synapse=0.01)
       probe_b = nengo.Probe(b, synapse=0.01)

   import nengo_distilled
   # build the model
   sim = nengo_distilled.Simulator(model)
   # run the model
   sim.run(10)

   # plot the results
   import matplotlib.pyplot as plt
   plt.plot(sim.trange(), sim.data[probe_a])
   plt.plot(sim.trange(), sim.data[probe_b])
   plt.show()

Installation
============

First, make sure ``nengo`` is installed by doing
``pip install nengo``.
You can also download and install Nengo manually
from `nengo/nengo <https://github.com/nengo/nengo>`_.

Now, install this by downloading the code
(``git clone https://github.com/nengo/nengo_distilled/``)
and doing ``python setup.py develop``
(or ``python setup.py develop --user``
if you want to install to your home directory).
