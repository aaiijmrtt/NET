# Getting Started

## Trying Different Layers

In this tutorial, we show you some of the different neural network layers that
you can experiment with using the net package. The initial setup remains the
same as in the previous tutorial.

		import random, numpy
		import net, data

		random.seed(0)
		numpy.random.seed(0)

		dataset = data.logical.datasetxor()
		split = int(0.8 * len(dataset))
		mytrainingset = dataset[: split]
		mytestingset = dataset[split: ]

		n_input = 2
		n_hidden = 6
		n_output = 1

In this tutorial, we shall try combinations of layers, cost functions and
hyperparameters. Let us try using a different hidden layer transfer function,
the hyperbolic tangent instead.

		mynet = net.Series()
		mynet.addlayer(net.Linear(n_input, n_hidden))
		mynet.addlayer(net.HyperbolicTangent(n_hidden))
		mynet.addlayer(net.Linear(n_hidden, n_output))
		mynet.addlayer(net.MeanSquared(n_output))

		mynet.applylearningrate(0.5)
		mynet.applyadaptivegradient()

The rest remains the same as in the previous tutorial.

		myoptimizer = net.Optimizer(mynet, mytrainingset, mytestingset)
		myoptimizer.train()
		error = myoptimizer.test()
		print(error)

You should get an error of 1.36432069e-06, which is much better. Now let us try
using a different cost function, the cross sigmoid instead.

		mynet = net.Series()
		mynet.addlayer(net.Linear(n_input, n_hidden))
		mynet.addlayer(net.HyperbolicTangent(n_hidden))
		mynet.addlayer(net.Linear(n_hidden, n_output))
		mynet.addlayer(net.CrossSigmoid(n_output))

		mynet.applylearningrate(0.15)
		mynet.applyvelocity(0.99)

You should get an error of 8.77023187e-07, which is even better. Finally, let
us try adding a fourth threshold layer.

		mynet = net.Series()
		mynet.addlayer(net.Linear(n_input, n_hidden))
		mynet.addlayer(net.HyperbolicTangent(n_hidden))
		mynet.addlayer(net.Linear(n_hidden, n_output))
		mynet.addlayer(net.Threshold(n_output))
		mynet.addlayer(net.MeanSquared(n_output))

		mynet.applylearningrate(0.15)
		mynet.applyvelocity(0.99)

You should get an error of 0.0. Now that is quite something. So far we have
dealt only with artificial datasets. In the next tutorial, we shall deal with
real world datasets.
