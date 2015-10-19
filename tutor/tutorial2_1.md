# Standard Datasets

## Iris

In this tutorial, we show you how to train a network on the Iris dataset using
the net package. The initial setup remains the same as in the previous tutorial.

		import numpy, random
		import net, data

		numpy.random.seed(0)
		random.seed(0)

The data package provides an interface to download and use some common datasets,
which are also available online.

		irisdataset = data.classification.datasetiris(True)
		split = int(0.8 * len(irisdataset))
		mytrainingset = irisdataset[: split]
		mytestingset = irisdataset[split: ]

We define a network with 2 sigmoidal hidden layers, of 10 units each. The input
and output layers have 4 and 3 units, respectively, corresponding to the 4
input features and 3 output classes.

		n_input = 4
		n_hidden1 = 10
		n_hidden2 = 10
		n_output = 3

		mynet = net.Series()
		mynet.addlayer(net.Linear(n_input, n_hidden1))
		mynet.addlayer(net.Sigmoid(n_hidden1))
		mynet.addlayer(net.Linear(n_hidden1, n_hidden2))
		mynet.addlayer(net.Sigmoid(n_hidden2))
		mynet.addlayer(net.Linear(n_hidden2, n_output))
		mynet.addlayer(net.MeanSquared(n_output))

		mynet.applylearningrate(0.5)

We define an optimizer and train for 40 epochs. We check the classification
error on the completion of training.

		myoptimizer = net.Optimizer(mynet, mytrainingset, mytestingset)
		myoptimizer.train(iterations = 40)
		error = myoptimizer.test(classification = True)
		print(error)

You should get an error of 0.0. Now that was pretty easy. We shall try more
challenging datasets in the next tutorial.
