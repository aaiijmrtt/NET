import numpy, visualize

class Multivariate(visualize.MultivariatePlot):

	def __init__(self, univariatelist, lowerlimits = None, upperlimits = None, stepsizes = None, functiondepictor = None, derivativedepictor = None):
		self.univariatelist = univariatelist
		visualize.MultivariatePlot.__init__(self, lowerlimits, upperlimits, stepsizes, functiondepictor, derivativedepictor)

	def minima(self):
		minimavector = numpy.empty((len(self.univariatelist), 1), dtype = float)
		for i in range(len(self.univariatelist)):
			minimavector[i][0] = self.univariatelist[i].minima()
		return minimavector

	def update(self):
		for univariate in self.univariatelist:
			univariate.update()

	def componentfunction(self, inputvector):
		outputvector = numpy.empty((len(self.univariatelist), 1), dtype = float)
		for i in range(len(self.univariatelist)):
			outputvector[i][0] = self.univariatelist[i].function(inputvector[i][0])
		return outputvector

	def function(self, inputvector):
		return self.combinefunction(self.componentfunction(inputvector))

	def componentderivative(self, inputvector):
		outputvector = numpy.empty((len(self.univariatelist), 1), dtype = float)
		for i in range(len(self.univariatelist)):
			outputvector[i][0] = self.univariatelist[i].derivative(inputvector[i][0])
		return outputvector

	def derivative(self, inputvector):
		return self.combinederivative(self.componentfunction(inputvector), self.componentderivative(inputvector))

class Sum(Multivariate):

	def __init__(self, univariatelist, lowerlimits = None, upperlimits = None, stepsizes = None, functiondepictor = None, derivativedepictor = None):
		Multivariate.__init__(self, univariatelist, lowerlimits, upperlimits, stepsizes, functiondepictor, derivativedepictor)
		self.combinefunction = numpy.sum
		self.combinederivative = lambda x, y: y

class L1Norm(Multivariate):

	def __init__(self, univariatelist, lowerlimits = None, upperlimits = None, stepsizes = None, functiondepictor = None, derivativedepictor = None):
		Multivariate.__init__(self, univariatelist, lowerlimits, upperlimits, stepsizes, functiondepictor, derivativedepictor)
		self.combinefunction = lambda x: numpy.sum(numpy.abs(x))
		self.combinederivative = numpy.vectorize(lambda x, y: -y if x < 0.0 else y)

class L2Norm(Multivariate):

	def __init__(self, univariatelist, lowerlimits = None, upperlimits = None, stepsizes = None, functiondepictor = None, derivativedepictor = None):
		Multivariate.__init__(self, univariatelist, lowerlimits, upperlimits, stepsizes, functiondepictor, derivativedepictor)
		self.combinefunction = lambda x: numpy.power(numpy.sum(numpy.square(x)), 0.5)
		self.combinederivative = lambda x, y: numpy.divide(numpy.multiply(x, y), numpy.sqrt(numpy.sum(numpy.square(x))))

class LPNorm(Multivariate):

	def __init__(self, univariatelist, power = None, lowerlimits = None, upperlimits = None, stepsizes = None, functiondepictor = None, derivativedepictor = None):
		Multivariate.__init__(self, univariatelist, lowerlimits, upperlimits, stepsizes, functiondepictor, derivativedepictor)
		self.power = power if power is not None else 1.0
		self.combinefunction = lambda x: numpy.power(numpy.sum(numpy.power(numpy.abs(x), self.power)), 1.0 / self.power)
		self.helper = numpy.vectorize(lambda w, x, y, z: - w * y * z if x < 0.0 else w * y * z)

	def combinederivative(self, functionvalues, derivativevalues):
		factor = numpy.power(numpy.sum(numpy.power(numpy.abs(functionvalues), self.power)), 1.0 / self.power - 1.0)
		powervalues = numpy.power(numpy.abs(functionvalues), self.power - 1.0)
		return self.helper(factor, functionvalues, derivativevalues, powervalues)
