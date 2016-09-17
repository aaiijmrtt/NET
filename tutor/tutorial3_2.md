# Boosting Performance

## Distributed Hyperparameter Optimization

In this tutorial, we show you how to use the distributed hyperparameter
optimizer. The initial setup and network configuration remains the same as in
the previous tutorial. To use the distributed hyperparameter optimizer, you must
first run dispy on all nodes. For our example we shall only use the local host.
Run `$ python dispynode.py -i localhost` from the terminal after installation.

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

The distributed hyperparameter optimizer assumes that the datasets exist in the
same directory and under the same name on all nodes. We shall write the training
and testing sets to directories in the local host. In reality, you shall have to
distribute these files across the nodes beforehand.

		trainpath = '~/data/trainingset'
		trainname = 'train'
		testpath = '~/data/testingset'
		testname = 'test'

		data.models.store(mytraining, trainpath, trainname)
		data.models.store(mytesting, testpath, testname)

We guess the range in which the best training hyperparameters shall lie.

		learningrates = [0.01, 0.005, 0.001]
		velocities = [0.1, 0.05, 0.01]
		hyper = [
			('applylearningrate', learningrates),
			('applyvelocity', velocities)
		]

We define a hyperparameter optimizer and train for 10 epochs. The system finds
the best combination of training hyperparameters for us, and returns the
classification error.

		myhoptimizer = net.DistributedHyperoptimizer(
			mynet,
			(trainpath, trainname),
			(testpath, testname)
		)
		besthyper, error = myhoptimizer.gridsearch(hyper, 1, 10, True)
		print(error)

You should still get an error of 0.04270463. However, every point in our grid
was searched in parallel. The estimated speedup should be as many times as there
are nodes on your cluster. In our example using the local host, this was 1. But
clearly this method scales far better than in the one in the previous tutorial.
Refer to the dispy documentation for details on how to set up your cluster.
