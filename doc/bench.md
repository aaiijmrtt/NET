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

			f(x) = fi(x) if i1 <= x <= i2

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

	* **Beta Probability Distribution**:

			p(x) = x ^ (p1 - 1) * (1 - x) ^ (p2 - 1) / integral_(0, 1)((x ^ (p1 - 1) * (1 - x) ^ (p2 - 1)) dx)

	* **Bernoulli Probability Distribution**:

			p(x) = C(p1, x) * p2 ^ x * (1 - p2) ^ (p1 - x)

	* **Gamma Probability Distribution**:

			p(x) = x ^ (p1 - 1) * exp(-x / p2) / ((p2 ^ p1) * Gamma(p1))

	* **Geometric Probability Distribution**:
	
			p(x) = p1 * (1 - p1) ^ (p2 - 1)

	* **Gaussian Probability Distribution**:

			p(x) = exp(-(x - p2) ^ 2 / (2 * p1 ^ 2)) / (2 * pi * p1 ^ 2) ^ 0.5

	* **Poisson Probability Distribution**:

			p(x) = p1 ^ x * exp(-x) / x!

	* **Uniform Probability Distribution**:

			p(x) = 1 / (p1 - p2)

	* **NoisyUnivariate**:

			wraps a noise probability distribution around a univariate function,
			additively or multiplicatively

5.	**Bench Marks**:

	* **Benchmark**:

			simplifies benchmarking modifiers

**Note**:

* \[x1, x2\] is used to denote vector concanetation.
