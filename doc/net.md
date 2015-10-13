#Classes

Each class (except Optimizers and Layer Modifiers) has feedforward and
backpropagate methods, which return the forwardfed output and the
backpropagated delta vector, respectively. Each object (except for Optimizer
and Layer Modifier objects) has previousinput and previousoutput datamembers.

1. **Layers**:

	* **Linear**:

			f(x) = W * x + b

		all modifiers are applicable to it

	* **Normalizer**:

			f(x)(i) = p1 * (x(i) - m(x(i))) / (v(x(i)) + e) ^ 0.5 + p2

		all modifiers are applicable to it

2. **Convolutions**:

	* **Convolutional**:

			f(x) = [W * conv(x) + b]

		all modifiers are applicable to it

	* **MaxPooling**:

			f(x) = [max(conv(x))]

	* **MinPooling**:

			f(x) = [min(conv(x))]

	* **AveragePooling**:

			f(x) = [avg(conv(x))]

3.	**Modifiers**:

	* **Decay**:

			w(t + 1) = w(t) - p1 / (1 + t * p2) * (dE(t) / dw(t))

	* **Velocity**:

			v(t + 1) = p1 * v(t) + p2 * (dE(t) / dw(t))
			w(t + 1) = w(t) - v(t)

	* **AdaptiveGain**:

			g(t + 1) = g(t) + p1 if (dE(t) / dw(t)) * (dE(t - 1) / dw(t - 1)) > 0
					(1 - p1) * g(t) otherwise
			w(t + 1) = w(t) - p2 * g(t) * dE(t) / dw(t)

	* **ResilientPropagation**:

			g(t + 1) = g(t) + p1 if (dE(t) / dw(t)) * (dE(t - 1) / dw(t - 1)) > 0
						(1 - p1) * g(t) otherwise
			w(t + 1) = w(t) - p2 * g(t) * sign(dE(t) / dw(t))

	* **AdaptiveGradient**:

			sw(t + 1) = sw(t) + (dE(t) / dw(t)) ^ 2
			w(t + 1) = w(t) - p / (sw(t + 1) + e) ^ 0.5 * (dE(t) / dw(t))

	* **RootMeanSquarePropagation**:

			msw(t + 1) = p1 * msw(t) + (1 - p1) * (dE(t) / dw(t)) ^ 2
			w(t + 1) = w(t) - p2 / (msw(t + 1) + e) ^ 0.5 * (dE(t) / dw(t))

	* **Regularization**:

			E = E + p * f(w)

	* **Dropout**:

			during training: x(i) = x(i) if random() > p
									0 otherwise
			during testing: x(i) = p * x(i)

4. **Connectors**:

	* **Split**:

			f(x) = [x, x ... x]

	* **MergeSum**:

			f([x1, x2 .. xn])(i) = sum_over_j(xj(i))

	* **MergeProduct**:

			f([x1, x2 .. xn])(i) = product_over_j(xj(i))

	* **Step**:

			f(x) = x

5. **Transfer Functions**:

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

	* **Threshold**:

			f(x)(i) = 1.0 if x(i) > 0.0
					= 0.0 otherwise

	* **StochasticThreshold**:

			f(x)(i) = 1.0 if x(i) > random()
					= 0.0 otherwise

6. **Error Functions**:

	* **MeanSquared**:

			f(y, o)(i) = 1 / 2 * (y(i) - o(i)) ^ 2

	* **CrossEntropy**:

			f(y, o)(i) = - (o(i) * log(y(i)) + (1 - o(i)) * log(1 - y(i)))

	* **NegativeLogLikelihood**:

			f(y, o)(i) = - o(i) * log(y(i))

	* **KullbackLeiblerDivergence**:

			f(y, o)(i) = o(i) * log(o(i) / y(i))

	* **CosineDistance**:

			f(y, o)(i) = - o(i) * y(i) / (sum_over_j(o(j) ^ 2) * sum_over_j(y(j) ^ 2)) ^ 0.5

	* **CrossSigmoid**:

			f(y, o)(i) = - (o(i) * log(g(y)(i)) + (1 - o(i)) * log(1 - g(y)(i)))
			g(y)(i) = 1 / (1 + exp(-y(i)))

	* **LogSoftMax**:

			f(y, o)(i) = - o(i) * log(g(y)(i))
			g(y)(i) = exp(y(i)) / sum_over_j(exp(y(j)))

7. **Perceptron**:

			f(x)(i) = 1.0 if g(x)(i) > random()
					= 0.0 otherwise
			g(x) = W * x + b

			all modifiers are applicable to it

8. **Hopfield Network**:

			f(x) = W * x where W(i)(j) = W(j)(i)
							and W(i)(i) = 0

			unsupervised pretraining: Hebbian Learning
			all modifiers are applicable to it

9. **RestrictedBoltzmannMachine**:

			f(x) = W' * g(x) + b2
			g(x)(i) = 1 / (1 + exp(-h(x)(i)))
			h(x) = W * x + b1

			unsupervised pretraining: Contrastive Divergence
			all modifiers are applicable to it

10. **AutoEncoder**:

			f(x) = W2 * g(x) + b2
			g(x)(i) = 1 / (1 + exp(-h(x)(i)))
			h(x) = W1 * x + b1

			unsupervised pretraining: Gradient Descent
			all modifiers are applicable to it

11. **Radial Basis Functions**:

			r(i) = (sum_over_j((x(j) - p1(i)(j)) ^ 2)) ^ 0.5

			unsupervised pretraining: K Means Clustering
			all modifiers are applicable to it

	* **GaussianRB**:

			f(x)(i) = exp(- r(i) ^ 2 / p2 ^ 2)

	* **MultiQuadraticRB**:

			f(x)(i) = (r(i) ^ 2 + p2 ^ 2) ^ p3

	* **InverseMultiQuadraticRB**:

			f(x)(i) = (r(i) ^ 2 + p2 ^ 2) ^ (-p3)

	* **ThinPlateSplineRB**:

			f(x)(i) = r(i) ^ 2 * log(r(i))

	* **CubicRB**:

			f(x)(i) = r(i) ^ 3

	* **LinearRB**:

			f(x)(i) = r(i)

12. **Self Organising Maps**:

			f(x)(i) = 1.0 if i = argmin(r(i))
					= 0.0 otherwise

			unsupervised pretraining: Competitive Learning
			all modifiers are applicable to it

	* **ManhattanSO**:

			r(i) = sum_over_j(|x(j) - w(i)(j)|)

	* **EuclideanSO**:

			r(i) = sum_over_j((x(j) - w(i)(j)) ^ 2) ^ 0.5

13. **Containers**:

	* **Series**:

			f(x) = fn( ... f2(f1(x)))

	* **Parallel**:

			f([x1, x2 ... xn]) = [f1(x1), f2(x2) ... fn(xn)]

	* **Recurrent**:

			F([h(t-1), x(t)]) = [h(t), f(x(t))]

14. **Optimizers**:

	* **Optimizer**: simplifies training and testing

	* **Hyperoptimizer**: optimizes hyperparameters, implements grid search and
Nelder-Meads algorithm

**Note**:

* x(i) indexing is used to denote the i-th space component of a vector x.
* x(t) indexing is used to denote the t-th time component of a vector x.
* \[x1, x2\] is used to denote vector concanetation.
