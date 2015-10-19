# Getting Started

## Creating Neural Networks

In this tutorial, we show you the simplest way to create a single layer
perceptron using the net package. First, we import all the packages we need.

		import random, numpy
		import net, data

A word of caution: following tutorials containing code that use random numbers
never guarantees reproducability. For that reason, we seed all random number
generators to begin with.

		random.seed(0)
		numpy.random.seed(0)

To generate the dataset, we use the data package. By default the dataset
contains 100 training samples. We split it 80-20 into the training and testing
sets.

		dataset = data.logical.datasetand()
		split = int(0.8 * len(dataset))
		mytrainingset = dataset[: split]
		mytestingset = dataset[split: ]

For this tutorial, we shall be training a single layer perceptron on the
two variable logical `and` function. Hence, we require the perceptron to have
2 input units and 1 output unit.

		n_input = 2
		n_output = 1

We then create a container. The series container is the most basic. All the
layers added to it are applied sequentially.

		mynet = net.Series()

We add the perceptron layer, with the required parameters.

		mynet.addlayer(net.Perceptron(n_input, n_output))

We shall be using the half mean squared error cost function for training.

		mynet.addlayer(net.MeanSquared(n_output))

We define an optimizer.

		myoptimizer = net.Optimizer(mynet, mytrainingset, mytestingset)

We use it to train on our dataset, using standard gradient descent.

		myoptimizer.train()

We check the error on completion.

		error = myoptimizer.test()
		print(error)

And that's it: we are done. If all goes well, you should get an error of 0.0,
which is as good as it gets. Congratulations, your network learned something.
We shall try a more challenging dataset in the next tutorial.
