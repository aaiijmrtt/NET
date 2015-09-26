#Classes

1. **Visualize**:

	* **UnivariatePlot**:

			plots the function and derivative values of a univariate function

	* **MultivariatePlot**:

			plots the partial function and partial derivative values of a
			multivariate function about a point

	* **HintonPlot**:

			plots the Hinton diagram of a matrix or vector

2. **Univariate Functions**:

	* **Linear**:

			f(x) ~ p * |x - x0|

	* **Quadratic**:

			f(x) ~ p * |x - x0| ^ 2

	* **Power**:

			f(x) ~ p1 * |x - x0| ^ p2

	* **Exponential**:

			f(x) ~ p1 * exp(p2 * (x - x0))

	* **Curve**:

			a piecewise univariate curve defined over intervals

3.	**Multivariate Functions**:

	* **Sum**:

			f([x1, x2 ... xn]) = sum_over_i(fi(xi))

	* **L1Norm**:

			f([x1, x2 ... xn]) = sum_over_i(|fi(xi)|)

	* **L2Norm**:

			f([x1, x2 ... xn]) = (sum_over_i(|fi(xi)| ^ 2)) ^ 0.5

	* **LPNorm**:

			f([x1, x2 ... xn]) = (sum_over_i(|fi(xi)| ^ p)) ^ (1 / p)

4. **Noise Functions**:

	* **Probability Distributions**:

			* Beta
			* Bernoulli
			* Gamma
			* Geometric
			* Gaussian
			* Poisson
			* Uniform

	* **NoisyUnivariate**:

			wraps a noise probability distribution around a univariate function,
			additively or multiplicatively

5.	**Bench Marks**:

	* **Benchmark**:

			simplifies benchmarking modifiers

**Note**:

* \[x1, x2\] is used to denote vector concanetation.
