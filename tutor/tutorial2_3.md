# Standard Datasets

## MNIST

In this tutorial, we show you how to train a network on the MNIST dataset using
the net package. The initial setup remains the same as in the previous tutorial.

		import numpy, random
		import net, data

		numpy.random.seed(0)
		random.seed(0)

		MNISTdataset = data.classification.datasetMNIST(True)
		split = int(0.8 * len(MNISTdataset))
		mytrainingset = MNISTdataset[: split]
		mytestingset = MNISTdataset[split: ]

We define a network with 3 hyperbolic tangent hidden layers, of 30 units each.
The input and output layers have 64 and 10 units, respectively, corresponding
to the 64 input features and 10 output classes.

		n_input = 64
		n_hidden1 = 30
		n_hidden2 = 30
		n_hidden3 = 30
		n_output = 10

		mynet = net.Series()
		mynet.addlayer(net.Linear(n_input, n_hidden1))
		mynet.addlayer(net.HyperbolicTangent(n_hidden1))
		mynet.addlayer(net.Linear(n_hidden1, n_hidden2))
		mynet.addlayer(net.HyperbolicTangent(n_hidden2))
		mynet.addlayer(net.Linear(n_hidden2, n_hidden3))
		mynet.addlayer(net.HyperbolicTangent(n_hidden3))
		mynet.addlayer(net.Linear(n_hidden3, n_output))
		mynet.addlayer(net.MeanSquared(n_output))

		mynet.applylearningrate(0.005)
		mynet.applyvelocity(0.05)

We define an optimizer and train for 10 epochs. We check the classification
error on the completion of training.

		myoptimizer = net.Optimizer(mynet, mytrainingset, mytestingset)
		myoptimizer.train(iterations = 10)
		error = myoptimizer.test(True)
		print(error)

You should get an error of 0.04270463. That is not bad for our first optical
character recognizer, but we can do better. We shall explore more exciting
architectures in the next tutorial.
