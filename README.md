# nengo_reference
An example backend implementation for nengo

This takes a neural model described using the Nengo framework, builds it 
into an actual neural simulation, and runs the simulation.  For example:

```python
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
    

import nengo_reference
# build the model
sim = nengo_reference.Simulator(model)
# run the model
sim.run(10)

# plot the results
import pylab
pylab.plot(sim.trange(), sim.data[probe_a])
pylab.plot(sim.trange(), sim.data[probe_b])
pylab.show()
```

# Installation

First, make sure ```nengo``` is installed by doing ```pip install nengo```.  You can also download nengo manually from https://github.com/nengo/nengo .

Now install this by downloading the code (```git clone https://github.com/ctn-waterloo/nengo_reference/```) and doing ```python setup.py develop --user```.
