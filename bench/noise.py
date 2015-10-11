import numpy
from . import univariate

class Beta:

	def __init__(self, alpha = None, beta = None, scale = None):
		self.alpha = alpha if alpha is not None else 1.0
		self.beta = beta if beta is not None else 1.0
		self.scale = scale if scale is not None else 1.0

	def sample(self):
		return self.scale * numpy.random.beta(self.alpha, self.beta)

class Bernoulli:

	def __init__(self, probability = None, trials = None, scale = None):
		self.probability = probability if probability is not None else 0.5
		self.trials = trials if trials is not None else 1.0
		self.scale = scale if scale is not None else 1.0

	def sample(self):
		return self.scale * numpy.random.binomial(self.trials, self.probability)

class Gamma:

	def __init__(self, exponent = None, scale = None):
		self.exponent = exponent if exponent is not None else 1.0
		self.scale = scale if scale is not None else 1.0

	def sample(self):
		return numpy.random.gamma(self.exponent, self.scale)

class Geometric:

	def __init__(self, probability = None, scale = None):
		self.probability = probability if probability is not None else 0.5
		self.scale = scale if scale is not None else 1.0

	def sample(self):
		return self.scale * numpy.random.geometric(self.probability)

class Gaussian:

	def __init__(self, mean = None, variance = None):
		self.mean = mean if mean is not None else 0.0
		self.variance = variance if variance is not None else 1.0

	def sample(self):
		return numpy.random.normal(self.mean, self.variance)

class Poisson:

	def __init__(self, lamda = None):
		self.lamda = lamda if lamda is not None else 1.0
		self.scale = scale if scale is not None else 1.0

	def sample(self):
		return self.scale * numpy.random.poisson(self.lamda)

class Uniform:

	def __init__(self, lowerlimit = None, upperlimit = None):
		self.lowerlimit = lowerlimit if lowerlimit is not None else 0.0
		self.upperlimit = upperlimit if upperlimit is not None else 1.0

	def sample(self):
		return numpy.random.uniform(self.lowerlimit, self.upperlimit)

# apply noise before piecing together as Curve
class NoisyUnivariate(univariate.Univariate):

	def __init__(self, pureunivariate, noise, additive = None):
		univariate.Univariate.__init__(self, pureunivariate.begin, pureunivariate.end, pureunivariate.value, pureunivariate.point, pureunivariate.shift, pureunivariate.lowerlimit, pureunivariate.upperlimit, pureunivariate.stepsize, pureunivariate.functiondepictor, pureunivariate.derivativedepictor)
		self.purefunction = pureunivariate.function
		self.purederivative = pureunivariate.derivative
		self.noise = noise
		self.additive = additive if additive is not None else True

	def function(self, point):
		value = self.purefunction(point)
		noise = self.noise.sample()
		return value + noise * (point - self.point) if self.additive else value * noise

	def derivative(self, point):
		value = self.purederivative(point)
		noise = self.noise.sample()
		return value + noise if self.additive else value * noise
