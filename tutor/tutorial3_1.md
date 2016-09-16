# Boosting Performance

## Hyperparameter Optimization

In this tutorial, we show you how to use the hyperparameter optimizer
effectively. The initial setup and network configuration remains the same as in
the previous tutorial.

		import numpy, random
		import net, data

		numpy.random.seed(0)
		random.seed(0)

		MNISTdataset = data.classification.datasetMNIST(True)
		split = int(0.8 * len(MNISTdataset))
		mytraining = MNISTdataset[: split]
		mytesting = MNISTdataset[split: ]

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

We guess the range in the best training hyperparameters shall lie.

		learningrates = [0.01, 0.005, 0.001]
		velocities = [0.1, 0.05, 0.01]
		hyperparameters = [
			('applylearningrate', learningrates),
			('applyvelocity', velocities)
		]

We define a hyperparameter optimizer and train for 10 epochs. The system finds
the best combination of training hyperparameters for us. We again check the
classification error on the completion of training.

		myhoptimizer = net.Hyperoptimizer(mynet, mytraining, mytesting)
		myhoptimizer.gridsearch(hyperparameters, 1, 10, True)
		error = myhoptimizer.test(True)
		print(error)

You should get an error of 0.04270463. This is the same error as in the pervious
tutorial, so how does this help? Last time, we magically knew the best
hyperparameters for our model. This time, we made a few conversative guesses in
likely ranges, and the hyperparameter optimizer did the rest for you. Of course,
this comes at the price of greater computation time, but that cannot be helped.
Or can it? Find out in the next tutorial.
