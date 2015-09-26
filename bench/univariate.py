import numpy, visualize

class Univariate(visualize.UnivariatePlot):

	def __init__(self, begin, end, value = None, point = None, shift = None):
		self.begin = begin
		self.end = end
		self.value = value
		self.point = point if point is not None else (self.begin + self.end) / 2.0
		self.shift = shift
		self.increasing = None

	def update(self):
		if self.shift is not None:
			self.point += self.shift.sample()

	def minima(self):
		if self.point < self.begin:
			return self.begin if self.increasing else self.end
		elif self.point > self.end:
			return self.end if self.increasing else self.beginning
		else:
			return self.point
		
class Linear(Univariate):

	def __init__(self, begin, end, value = None, point = None, parameter = None):
		Univariate.__init__(self, begin, end, value, point)
		self.parameter = parameter if parameter is not None else 1.0
		self.function = lambda x: self.parameter * numpy.abs(x - self.point) + self.value
		self.derivative = lambda x: - self.parameter if x < self.point else self.parameter
		self.increasing = self.parameter > 0.0

class Quadratic(Univariate):

	def __init__(self, begin, end, value = None, point = None, parameter = None):
		Univariate.__init__(self, begin, end, value, point)
		self.parameter = parameter if parameter is not None else 1.0
		self.function = lambda x: self.parameter * numpy.square(x - self.point) + self.value
		self.derivative = lambda x: 2.0 * self.parameter * (x - self.point)
		self.increasing = self.parameter > 0.0

class Power(Univariate):

	def __init__(self, begin, end, value = None, point = None, parameter = None, power = None):
		Univariate.__init__(self, begin, end, value, point)
		self.parameter = parameter if parameter is not None else 1.0
		self.power = power if power is not None else 1.0
		self.function = lambda x: self.parameter * numpy.power(numpy.abs(x - self.point), self.power) + self.value
		self.derivative = lambda x: self.parameter * self.power * numpy.power(x - self.point, self.power - 1.0)
		self.increasing = self.parameter > 0.0

class Exponential(Univariate):

	def __init__(self, begin, end, value = None, point = None, parameter1 = None, parameter2 = None):
		Univariate.__init__(self, begin, end, value, point)
		self.parameter1 = parameter1 if parameter1 is not None else 1.0
		self.parameter2 = parameter2 if parameter2 is not None else 1.0
		self.function = lambda x: self.parameter1 * numpy.exp(self.parameter2 * (x - self.point)) + self.value - self.parameter1
		self.derivative = lambda x: self.parameter1 * self.parameter2 * numpy.exp(self.parameter2 * (x - self.point))
		self.increasing = self.parameter1 * self.parameter2 > 0.0

class Curve(visualize.UnivariatePlot):

	def __init__(self, functionlist):
		self.functionlist, self.rangelist = zip(*sorted(zip(functionlist, [(function.begin, function.end) for function in functionlist]), key = lambda x: x[1][0]))
		self.begin = self.rangelist[0][0]
		self.end = self.rangelist[-1][1]

	def function(self, point):
		for function in self.functionlist:
			if function.begin <= point < function.end:
				return function.function(point)
		return numpy.nan

	def derivative(self, point):
		for function in self.functionlist:
			if function.begin <= point < function.end:
				return function.derivative(point)
		return numpy.nan

	def minima(self):
		minimum = float('inf')
		point = None
		for function in self.functionlist:
			minima = function.minima()
			value = function.function(minima)
			if value < minimum:
				point = minima
				minimum = value
		return point

	def update(self):
		for function in self.functionlist:
			function.update()
