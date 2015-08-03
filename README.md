#NET

**Neural Networks in Python**

Putting Neural Networks together should be easier than it usually is. The code
in this repository presents simple Python snippets designed to make prototyping
neural network architectures easy.

**Note**:

* Requires Python.
* Requires Numpy.

##Classes

Each class has feedforward, backpropagate, updateweights and cleardeltas methods.
Each class has previousinput and previousoutput datamembers.

1. **Layers**:

	* **Linear**:

			f(x) = W * x + b

2. **Transfer Functions**:

	* **Sigmoid**:

			f(x)(i) = 1 / (1 + exp(-x(i)))

	* **HyperbolicTangent**:

			f(x)(i) = (exp(x(i)) - exp(-x(i))) / (exp(x(i)) + exp(-x(i)))

	* **RectifiedLinearUnit**:

			f(x)(i) = x(i) if x(i) > 0
					= 0 otherwise

	* **ParametricRectifiedLinearUnit**:

			f(x)(i) = x(i) if x(i) > 0
					= p * x(i) otherwise

	* **HardHyperbolicTangent**:

			f(x)(i) = x(i) if |x(i)| < 1
					= |x(i)| / x(i) otherwise

	* **HardShrink**:

			f(x)(i) = x(i) if |x(i)| > p
					= 0 otherwise

	* **SoftShrink**:

			f(x)(i) = x(i) - |x(i)| / x(i) * p if |x(i)| > p
					= 0 otherwise

	* **SoftPlus**:

			f(x)(i) = 1 / p * log(1 + exp(p * x(i)))

	* **SoftMax**:

			f(x)(i) = exp(x(i)) / sum_over_j(exp(x(j)))

3. **Error Functions**:

	* **MeanSquared**:

			f(y, o)(i) = 1 / 2 * (y(i) - o(i)) ^ 2

4. **Containers**:

	* **Series**:

			f(x) = fn( ... f2(f1(x)))

	* **Parallel**:

			f([x1, x2 ... xn]) = [f1(x1), f2(x2) ... fn(xn)]

**Note**:

* (i) indexing is used to denote the i-th component of a vector.
* \[x1, x2\] is used to denote vector concanetation.

##Sample Code

1. **Putting together a simple network**:

		# Input -> Linear -> Sigmoid -> Parallel Linear -> HyperbolicTangent -> Linear -> Output

		import net

		n_input = 2
		n_hidden1 = 6
		n_hidden2 = 6
		n_output = 1

		myseriesnet = net.Series()
		myseriesnet.addlayer(net.Linear(n_input, n_hidden1))
		myseriesnet.addlayer(net.Sigmoid(n_hidden1))

		myparallelnet = net.Parallel()
		myparallelnet.addlayer(net.Linear(n_hidden1 / 2, n_hidden2 / 2))
		myparallelnet.addlayer(net.Linear(n_hidden1 / 2, n_hidden2 / 2))

		myseriesnet.addlayer(myparallelnet)
		myseriesnet.addlayer(net.HyperbolicTangent(n_hidden2))
		myseriesnet.addlayer(net.Linear(n_hidden2, n_output))
		myseriesnet.addlayer(net.MeanSquared(n_output))

2. **Training on a simple function**:

		# f(0, 0) = 0
		# f(0, 1) = 1
		# f(1, 0) = 1
		# f(1, 1) = 0

		import numpy

		x = numpy.zeros((n_input, 1), dtype = float)
		y = numpy.zeros((n_output, 1), dtype = float)

		for i in range(10004):

			if i % 4 == 0:
				x[0][0] = 0.0
				x[1][0] = 0.0
				y[0][0] = 0.0
			elif i % 4 == 1:
				x[0][0] = 0.0
				x[1][0] = 1.0
				y[0][0] = 1.0
			elif i % 4 == 2:
				x[0][0] = 1.0
				x[1][0] = 0.0
				y[0][0] = 1.0
			else:
				x[0][0] = 1.0
				x[1][0] = 1.0
				y[0][0] = 0.0

			if i < 10000:
				myseriesnet.feedforward(x)
				myseriesnet.backpropagate(y)
				myseriesnet.updateweights()
			else:
				print x, myseriesnet.feedforward(x)

**Note**:

The overparameterization is a result of the need to display basic functionality,
while the overfitting is a result of the need for the data to be simple and for
the optimization to converge reliably. This reflects my opinion that first
examples should be unnecessarily simple and work unreasonably well. Run the
example from the parent directory.
