#NET

**Neural Networks in Python**

Putting Neural Networks together should be easier than it usually is. The code
in this repository presents simple Python snippets designed to make prototyping
neural network architectures quick.

**Note**:

* Requires Python.
* Requires Numpy.

##Sample Code

1. **Putting Together a Simple Network**:

		# Input -> Linear -> HyperbolicTangent -> Linear -> MeanSquared (Error) -> Output

		import net

		n_input = 2
		n_hidden = 6
		n_output = 1

		myseriesnet = net.Series()

		myseriesnet.addlayer(net.Linear(n_input, n_hidden))
		myseriesnet.addlayer(net.HyperbolicTangent(n_hidden))
		myseriesnet.addlayer(net.Linear(n_hidden, n_output))
		myseriesnet.addlayer(net.MeanSquared(n_output))

		myseriesnet.applyvelocity(0.9)
		myseriesnet.applyregularization()

2. **Creating a Toy Dataset**:

		# f(0, 0) = 0
		# f(0, 1) = 1
		# f(1, 0) = 1
		# f(1, 1) = 0

		import numpy

		mytrainingset = list()

		for i in range(500):

			x = numpy.zeros((n_input, 1), dtype = float)
			y = numpy.zeros((n_output, 1), dtype = float)

			if i % 4 == 0:
				mytrainingset.append((x, y))
			elif i % 4 == 1:
				x[1][0] = 1.0
				y[0][0] = 1.0
				mytrainingset.append((x, y))
			elif i % 4 == 2:
				x[0][0] = 1.0
				y[0][0] = 1.0
				mytrainingset.append((x, y))
			else:
				x[0][0] = 1.0
				x[1][0] = 1.0
				mytrainingset.append((x, y))

		mytestingset = mytrainingset

3. **Training and Testing the Network**:

		myoptimizer = net.Optimizer(myseriesnet, mytrainingset, mytestingset, lambda x, y: 0.5 * (x - y) ** 2)
		myoptimizer.train()

		print "error:", myoptimizer.test()
