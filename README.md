#NET

**Neural Networks in Python**

Putting Neural Networks together should be easier than it usually is. The code
in this repository presents simple Python snippets designed to make prototyping
neural network architectures quick.

**Note**:

* Requires Python.
* Requires Numpy.

##Classes

Each class (except Optimizer) has feedforward and backpropagate methods, which
return the forwardfed output and the backpropagated delta vector, respectively.
Each object (except of Optimizer objects) has previousinput and previousoutput
datamembers.

1. **Layers**:

	* **Linear**:

			f(x) = W * x + b

	* **Split**:

			f(x) = [x, x ... x]

	* **MergeSum**:

			f([x1, x2 .. xn])(i) = sum_over_j(xj(i))

	* **MergeProduct**:

			f([x1, x2 .. xn])(i) = product_over_j(xj(i))

	* **Velocity**: modifier applicable to training of Linear layers

	* **Regularization**: modifier applicable to training of Linear layers

	* **Dropout**: modifier applicable to training of Linear layers

2. **Transfer Functions**:

	* **ShiftScale**:

			f(x)(i) = p1 * x(i) + p2

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

5. **Optimizers**:

	* **Optimizer**: simplifies training and testing

**Note**:

* x(i) indexing is used to denote the i-th component of a vector x.
* \[x1, x2\] is used to denote vector concanetation.

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
				x[0][0] = 0.0
				x[1][0] = 0.0
				y[0][0] = 0.0
				mytrainingset.append((x, y))
			elif i % 4 == 1:
				x[0][0] = 0.0
				x[1][0] = 1.0
				y[0][0] = 1.0
				mytrainingset.append((x, y))
			elif i % 4 == 2:
				x[0][0] = 1.0
				x[1][0] = 0.0
				y[0][0] = 1.0
				mytrainingset.append((x, y))
			else:
				x[0][0] = 1.0
				x[1][0] = 1.0
				y[0][0] = 0.0
				mytrainingset.append((x, y))

		mytestingset = mytrainingset

3. **Training and Testing the Network**:

		myoptimizer = net.Optimizer(myseriesnet, mytrainingset, mytestingset, lambda x, y: 0.5 * (x - y) ** 2)
		myoptimizer.train()

		print "error:", myoptimizer.test()

**Note**:

The module must be properly pointed to when running your code. Hint: Simply run
from the parent directory.
