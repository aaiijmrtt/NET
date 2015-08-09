#Classes

Each class (except Optimizer) has feedforward and backpropagate methods, which
return the forwardfed output and the backpropagated delta vector, respectively.
Each object (except for Optimizer objects) has previousinput and previousoutput
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

	* **Step**:

			f(x) = x

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

	* **Recurrent**

5. **Optimizers**:

	* **Optimizer**: simplifies training and testing

**Note**:

* x(i) indexing is used to denote the i-th component of a vector x.
* \[x1, x2\] is used to denote vector concanetation.
