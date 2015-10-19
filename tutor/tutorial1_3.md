# Getting Started

## Training Neural Networks

In this tutorial, we show you different ways to optimize the training of a
neural network using the net package. The initial setup remains the same as in
the previous tutorial.

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

		mynet = net.Series()
		mynet.addlayer(net.Linear(n_input, n_hidden))
		mynet.addlayer(net.Sigmoid(n_hidden))
		mynet.addlayer(net.Linear(n_hidden, n_output))
		mynet.addlayer(net.MeanSquared(n_output))

As before, we define an optimizer, and use it for training.

		myoptimizer = net.Optimizer(mynet, mytrainingset, mytestingset)
		myoptimizer.train()

As before, we check the error on completion.

		error = myoptimizer.test()
		print(error)

Our baseline error is 0.13727146. Let us play around with the training code,
and see if we can improve on it. An evil 'optimization' would be to train for
a large number of iterations.

		myoptimizer = net.Optimizer(mynet, mytrainingset, mytestingset)
		myoptimizer.train(iterations = 50)

You should get an error of 0.00561724. But this takes too long, and does not
always work. Let us try being a little more imaginative with the training
algorithms. How about learning, with adaptive gradients?

		mynet.applylearningrate(0.2)
		mynet.applyadaptivegradient()
		myoptimizer = net.Optimizer(mynet, mytrainingset, mytestingset)
		myoptimizer.train()

You should get an error of 0.12187924. How about learning, with velocity?

		mynet.applylearningrate(0.1)
		mynet.applyvelocity(0.99)
		myoptimizer = net.Optimizer(mynet, mytrainingset, mytestingset)
		myoptimizer.train()

You should get an error of 0.05529851. While this is much better, it seems that
we might have reached the limit of this model. We shall try other models in the
next tutorial.
