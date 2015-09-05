#Classes

Each class (except Optimizers and Layer Modifiers) has feedforward and
backpropagate methods, which return the forwardfed output and the
backpropagated delta vector, respectively. Each object (except for Optimizer
and Layer Modifier objects) has previousinput and previousoutput datamembers.

1. **Layers**:

	* **Linear**:

			f(x) = W * x + b

	* **Normalizer**:

			f(x)(i) = p1 * (x(i) - m(x(i))) / (v(x(i)) + e) ^ 0.5 + p2

	* **Modifier**:

		* **Velocity**: implements momentum based gradient descent, accelerates
learning

		* **Regularization**: implements regularization, prevents overfitting

		* **Dropout**: implements dropout, prevents overfitting

2. **Connectors**:

	* **Split**:

			f(x) = [x, x ... x]

	* **MergeSum**:

			f([x1, x2 .. xn])(i) = sum_over_j(xj(i))

	* **MergeProduct**:

			f([x1, x2 .. xn])(i) = product_over_j(xj(i))

	* **Step**:

			f(x) = x

3. **Transfer Functions**:

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

	* **SoftSign**:

			f(x)(i) = 1 / (1 + |x(i)|)

	* **SoftMax**:

			f(x)(i) = exp(x(i)) / sum_over_j(exp(x(j)))

4. **Error Functions**:

	* **MeanSquared**:

			f(y, o)(i) = 1 / 2 * (y(i) - o(i)) ^ 2

	* **CrossEntropy**:

			f(y, o)(i) = - (o(i) * log(y(i)) + (1 - o(i)) * log(1 - y(i)))

	* **NegativeLogLikelihood**:

			f(y, o)(i) = - o(i) * log(y(i))

	* **CrossSigmoid**: implements composition of Sigmoid Transfer Function and
CrossEntropy Error Function

	* **LogSoftMax**: implements composition of SoftMax Transfer Function and
NegativeLogLikelihood Error Function

5. **Containers**:

	* **Series**:

			f(x) = fn( ... f2(f1(x)))

	* **Parallel**:

			f([x1, x2 ... xn]) = [f1(x1), f2(x2) ... fn(xn)]

	* **Recurrent**: implements time recurrence

6. **Optimizers**:

	* **Optimizer**: simplifies training and testing

	* **Hyperoptimizer**: optimizes hyperparameters, implements grid search and
Nelder-Meads algorithm

**Note**:

* x(i) indexing is used to denote the i-th component of a vector x.
* \[x1, x2\] is used to denote vector concanetation.
