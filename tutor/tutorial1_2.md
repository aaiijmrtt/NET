# Getting Started

## Adding Multiple Layers

In this tutorial, we show you the simplest way to create multiple layer
neural networks using the net package. As before, we import all the packages we
need, seed all random number generators, and generate a dataset.

		import random, numpy
		import net, data

		random.seed(0)
		numpy.random.seed(0)

		dataset = data.logical.datasetxor()
		split = int(0.8 * len(dataset))
		mytrainingset = dataset[: split]
		mytestingset = dataset[split: ]

If you followed the previous tutorial, replacing the `and` dataset with `xor`
would leave you with an error of 0.25. That is because the perceptron can only
model linearly separable functions. So we need a better model. For this
tutorial, we shall be creating a three layer neural network: a linear input
layer, followed by a nonlinear sigmoid hidden layer, followed by another linear
output layer. The hidden layer will have 6 units.

		n_input = 2
		n_hidden = 6
		n_output = 1

		mynet = net.Series()
		mynet.addlayer(net.Linear(n_input, n_hidden))
		mynet.addlayer(net.Sigmoid(n_hidden))
		mynet.addlayer(net.Linear(n_hidden, n_output))

The rest remains the same as in the previous tutorial.

		mynet.addlayer(net.MeanSquared(n_output))

		myoptimizer = net.Optimizer(mynet, mytrainingset, mytestingset)
		myoptimizer.train()
		error = myoptimizer.test()
		print(error)

If all goes well, you should get an error of 0.13727146. That is a bit better
than 0.25, but it still isn't all that great. We shall improve on that in the
next tutorial.
