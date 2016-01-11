# Standard Datasets

## Boston Housing

In this tutorial, we show you how to train a network on the Boston Housing
dataset using the net package. The initial setup remains the same as in the
previous tutorial.

		import numpy, random
		import net, data

		numpy.random.seed(0)
		random.seed(0)

We shall normalize the dataset, before attempting regression.

		bostonhousingdataset = data.regression.datasetbostonhousing(True)
		bostonhousingdataset = data.tools.pairednormalization(bostonhousingdataset)
		split = int(0.8 * len(bostonhousingdataset))
		mytrainingset = bostonhousingdataset[: split]
		mytestingset = bostonhousingdataset[split: ]

We define a network with 2 hyperbolic tangent hidden layers, of 30 units each.
The input and output layers have 13 and 1 units, respectively, corresponding to
the 13 input features and 1 output classes.

		n_input = 13
		n_hidden1 = 30
		n_hidden2 = 30
		n_output = 1

		mynet = net.Series()
		mynet.addlayer(net.Linear(n_input, n_hidden1))
		mynet.addlayer(net.HyperbolicTangent(n_hidden1))
		mynet.addlayer(net.Linear(n_hidden1, n_hidden2))
		mynet.addlayer(net.HyperbolicTangent(n_hidden2))
		mynet.addlayer(net.Linear(n_hidden2, n_output))
		mynet.addlayer(net.MeanSquared(n_output))

		mynet.applylearningrate(0.07)
		mynet.applyadaptivegradient()

We define an optimizer and train for 10 epochs. We check the regression error
on the completion of training.

		myoptimizer = net.Optimizer(mynet, mytrainingset, mytestingset)
		myoptimizer.train(iterations = 10)
		error = myoptimizer.test()
		print(error)

You should get an error of 0.00365975.
