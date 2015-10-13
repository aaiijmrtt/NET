'''
	Module containing Univariate Function Noise Generators.
	Classes embody Stochastic Noise Distributions,
	combined additively or multiplicatively with function gradient.
'''
import numpy
from . import univariate

class Beta:
	'''
		Beta Probability Distribution Function
		Mathematically, p(x) = x ^ (p1 - 1) * (1 - x) ^ (p2 - 1) / integral_(0, 1)((x ^ (p1 - 1) * (1 - x) ^ (p2 - 1)) dx)
	'''
	def __init__(self, alpha = None, beta = None, scale = None):
		'''
			Constructor
			: param alpha : p1, as given in its mathematical expression
			: param beta : p2, as given in its mathematical expression
			: param scale : factor by which random sample is scaled
		'''
		self.alpha = alpha if alpha is not None else 1.0
		self.beta = beta if beta is not None else 1.0
		self.scale = scale if scale is not None else 1.0

	def sample(self):
		'''
			Method to return a random sample from the probability distribution
		'''
		return self.scale * numpy.random.beta(self.alpha, self.beta)

class Bernoulli:
	'''
		Bernoulli Probability Distribution Function
		Mathematically, p(x) = C(p1, x) * p2 ^ x * (1 - p2) ^ (p1 - x)
	'''
	def __init__(self, probability = None, trials = None, scale = None):
		'''
			Constructor
			: param probability : p2, as given in its mathematical expression
			: param trials : p1, as given in its mathematical expression
			: param scale : factor by which random sample is scaled
		'''
		self.probability = probability if probability is not None else 0.5
		self.trials = trials if trials is not None else 1.0
		self.scale = scale if scale is not None else 1.0

	def sample(self):
		'''
			Method to return a random sample from the probability distribution
		'''
		return self.scale * numpy.random.binomial(self.trials, self.probability)

class Gamma:
	'''
		Gamma Probability Distribution Function
		Mathematically, p(x) = x ^ (p1 - 1) * exp(-x / p2) / ((p2 ^ p1) * Gamma(p1))
	'''
	def __init__(self, exponent = None, scale = None):
		'''
			Constructor
			: param exponent : p1, as given in its mathematical expression
			: param scale : p2, as given in its mathematical expression
		'''
		self.exponent = exponent if exponent is not None else 1.0
		self.scale = scale if scale is not None else 1.0

	def sample(self):
		'''
			Method to return a random sample from the probability distribution
		'''
		return numpy.random.gamma(self.exponent, self.scale)

class Geometric:
	'''
		Geometric Probability Distribution Function
		Mathematically, p(x) = p1 * (1 - p1) ^ (x - 1)
	'''
	def __init__(self, probability = None, scale = None):
		'''
			Constructor
			: param probability : p1, as given in its mathematical expression
			: param scale : factor by which random sample is scaled
		'''
		self.probability = probability if probability is not None else 0.5
		self.scale = scale if scale is not None else 1.0

	def sample(self):
		'''
			Method to return a random sample from the probability distribution
		'''
		return self.scale * numpy.random.geometric(self.probability)

class Gaussian:
	'''
		Gaussian Probability Distribution Function
		Mathematically, p(x) = exp(-(x - p1) ^ 2 / (2 * p2 ^ 2)) / (2 * pi * p2 ^ 2) ^ 0.5
	'''
	def __init__(self, mean = None, variance = None):
		'''
			Constructor
			: param mean : p1, as given in its mathematical expression
			: param variance : p2, as given in its mathematical expression
		'''
		self.mean = mean if mean is not None else 0.0
		self.variance = variance if variance is not None else 1.0

	def sample(self):
		'''
			Method to return a random sample from the probability distribution
		'''
		return numpy.random.normal(self.mean, self.variance)

class Poisson:
	'''
		Poisson Probability Distribution Function
		Mathematically, p(x) = p1 ^ x * exp(-x) / x!
	'''
	def __init__(self, lamda = None, scale = None):
		'''
			Constructor
			: param lamda : p1, as given in its mathematical expression
			: param scale : factor by which random sample is scaled
		'''
		self.lamda = lamda if lamda is not None else 1.0
		self.scale = scale if scale is not None else 1.0

	def sample(self):
		'''
			Method to return a random sample from the probability distribution
		'''
		return self.scale * numpy.random.poisson(self.lamda)

class Uniform:
	'''
		Uniform Probability Distribution Function
		Mathematically, p(x) = 1 / (p1 - p2)
	'''
	def __init__(self, lowerlimit = None, upperlimit = None):
		'''
			Constructor
			: param upperlimit : p2, as given in its mathematical expression
			: param lowerlimit : p1, as given in its mathematical expression
		'''
		self.lowerlimit = lowerlimit if lowerlimit is not None else 0.0
		self.upperlimit = upperlimit if upperlimit is not None else 1.0

	def sample(self):
		'''
			Method to return a random sample from the probability distribution
		'''
		return numpy.random.uniform(self.lowerlimit, self.upperlimit)

# apply noise before piecing together as Curve
class NoisyUnivariate(univariate.Univariate):
	'''
		Univariate Function Stochastic Noise Wrapper
	'''
	def __init__(self, pureunivariate, noise, additive = None):
		'''
			Constructor
			: param pureunivariate : pure univariate function to wrap with noises
			: param noise : noise probability distribution function
			: param additive : additive or multiplicative wrapper
		'''
		univariate.Univariate.__init__(self, pureunivariate.begin, pureunivariate.end, pureunivariate.value, pureunivariate.point, pureunivariate.shift, pureunivariate.lowerlimit, pureunivariate.upperlimit, pureunivariate.stepsize, pureunivariate.functiondepictor, pureunivariate.derivativedepictor)
		self.purefunction = pureunivariate.function
		self.purederivative = pureunivariate.derivative
		self.noise = noise
		self.additive = additive if additive is not None else True

	def function(self, point):
		'''
			Method to evaluate noisy univariate function
			: param inputvector : point of evaluation in parameter space
			: returns : evaluated function value at point in parameter space
		'''
		value = self.purefunction(point)
		noise = self.noise.sample()
		return value + noise * (point - self.point) if self.additive else value * noise

	def derivative(self, point):
		'''
			Method to evaluate noisy univariate derivative
			: param inputvector : point of evaluation in parameter space
			: returns : evaluated derivative value at point in parameter space
		'''
		value = self.purederivative(point)
		noise = self.noise.sample()
		return value + noise if self.additive else value * noise
