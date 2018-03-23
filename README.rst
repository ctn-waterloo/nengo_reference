=========================
A distilled Nengo backend
=========================

This is an example backend implementation for Nengo.

It does not require any additional dependencies;
instead, this backend is the
`reference backend <https://github.com/nengo/nengo>`_
distilled into the essential parts.
As such, it is designed to be
simpler, easier to understand,
and easier to debug.
If you're interested in writing your own Nengo backend,
then this implementation is likely a better
starting point than the reference backend.

`nengo_distilled` takes a neural model described
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

The easiest way to install is to use ``pip``.

.. code-block:: bash

   pip install nengo-distilled

If that doesn't work, then try
installing ``nengo`` manually,
using the instructions at
`nengo/nengo <https://github.com/nengo/nengo>`_.
Then, try ``pip install nengo-distilled`` again.
If that doesn't work, try a develop installation.

Developer installation
----------------------

If you want to make changes to ``nengo_distilled``,
then do the following.

.. code-block:: bash

   git clone https://github.com/nengo/nengo-distilled/
   cd nengo-distilled
   python setup.py develop --user

If you’re using a ``virtualenv`` (recommended!)
then you can omit the ``--user`` flag.
