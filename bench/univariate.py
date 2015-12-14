'''
	Module containing Univariate Function Generators.
	Classes embody Univariate Functions.
'''
import numpy
from . import visualize

class Univariate(visualize.UnivariatePlot):
	'''
		Base Class for Univariate Functions
	'''
	def __init__(self, begin, end, value = None, point = None, shift = None, lowerlimit = None, upperlimit = None, stepsize = None, functiondepictor = None, derivativedepictor = None):
		'''
			Constructor
			: param begin : lower limit of function domain in reals
			: param end : upper limit of function domain in reals
			: param value : value of function at point
			: param point : function minima in unrestricted domain of reals
			: param shift : parameter determining non stationarity of function minima
			: param lowerlimit : lower limit used to plot along univariate function axis
			: param upperlimit : upper limit used to plot along univariate function axis
			: param stepsize : step size used to plot along univariate function axis
			: param functiondepictor : depictor used to plot function
			: param derivativedepictor : depictor used to plot function derivative
		'''
		self.begin = begin
		self.end = end
		self.value = value if value is not None else 0.0
		self.point = point if point is not None else (self.begin + self.end) / 2.0
		self.shift = shift
		self.increasing = None
		visualize.UnivariatePlot.__init__(self, lowerlimit, upperlimit, stepsize, functiondepictor, derivativedepictor)

	def update(self):
		'''
			Method to update non stationary minima of univariate function
		'''
		if self.shift is not None:
			self.point += self.shift.sample()

	def minima(self):
		'''
			Method to calculate true mimima of univariate function
			: returns : point corresponding to minima of univariate function
		'''
		if self.point < self.begin:
			return self.begin if self.increasing else self.end
		elif self.point > self.end:
			return self.end if self.increasing else self.beginning
		else:
			return self.point

class Linear(Univariate):
	'''
		Linear Univariate Function
		Mathematically, f(x) ~ p * |x - x0|
	'''
	def __init__(self, begin, end, value = None, point = None, shift = None, parameter = None, lowerlimit = None, upperlimit = None, stepsize = None, functiondepictor = None, derivativedepictor = None):
		'''
			Constructor
			: param begin : lower limit of function domain in reals
			: param end : upper limit of function domain in reals
			: param value : value of function at point
			: param point : x0, as given in its mathematical expression
			: param shift : parameter determining non stationarity of function minima
			: param parameter : p, as given in its mathematical expression
			: param lowerlimit : lower limit used to plot along univariate function axis
			: param upperlimit : upper limit used to plot along univariate function axis
			: param stepsize : step size used to plot along univariate function axis
			: param functiondepictor : depictor used to plot function
			: param derivativedepictor : depictor used to plot function derivative
		'''
		Univariate.__init__(self, begin, end, value, point, shift, lowerlimit, upperlimit, stepsize, functiondepictor, derivativedepictor)
		self.parameter = parameter if parameter is not None else 1.0
		self.function = lambda x: self.parameter * numpy.abs(x - self.point) + self.value
		self.derivative = lambda x: - self.parameter if x < self.point else self.parameter
		self.increasing = self.parameter > 0.0

class Quadratic(Univariate):
	'''
		Quadratic Univariate Function
		Mathematically, f(x) ~ p * |x - x0| ^ 2
	'''
	def __init__(self, begin, end, value = None, point = None, shift = None, parameter = None, lowerlimit = None, upperlimit = None, stepsize = None, functiondepictor = None, derivativedepictor = None):
		'''
			Constructor
			: param begin : lower limit of function domain in reals
			: param end : upper limit of function domain in reals
			: param value : value of function at point
			: param point : x0, as given in its mathematical expression
			: param shift : parameter determining non stationarity of function minima
			: param parameter : p, as given in its mathematical expression
			: param lowerlimit : lower limit used to plot along univariate function axis
			: param upperlimit : upper limit used to plot along univariate function axis
			: param stepsize : step size used to plot along univariate function axis
			: param functiondepictor : depictor used to plot function
			: param derivativedepictor : depictor used to plot function derivative
		'''
		Univariate.__init__(self, begin, end, value, point, shift, lowerlimit, upperlimit, stepsize, functiondepictor, derivativedepictor)
		self.parameter = parameter if parameter is not None else 1.0
		self.function = lambda x: self.parameter * numpy.square(x - self.point) + self.value
		self.derivative = lambda x: 2.0 * self.parameter * (x - self.point)
		self.increasing = self.parameter > 0.0

class Power(Univariate):
	'''
		Power Univariate Function
		Mathematically, f(x) ~ p1 * |x - x0| ^ p2
	'''
	def __init__(self, begin, end, value = None, point = None, shift = None, parameter = None, power = None, lowerlimit = None, upperlimit = None, stepsize = None, functiondepictor = None, derivativedepictor = None):
		'''
			Constructor
			: param begin : lower limit of function domain in reals
			: param end : upper limit of function domain in reals
			: param value : value of function at point
			: param point : x0, as given in its mathematical expression
			: param shift : parameter determining non stationarity of function minima
			: param parameter : p1, as given in its mathematical expression
			: param power : p2, as given in its mathematical expression
			: param lowerlimit : lower limit used to plot along univariate function axis
			: param upperlimit : upper limit used to plot along univariate function axis
			: param stepsize : step size used to plot along univariate function axis
			: param functiondepictor : depictor used to plot function
			: param derivativedepictor : depictor used to plot function derivative
		'''
		Univariate.__init__(self, begin, end, value, point, shift, lowerlimit, upperlimit, stepsize, functiondepictor, derivativedepictor)
		self.parameter = parameter if parameter is not None else 1.0
		self.power = power if power is not None else 1.0
		self.function = lambda x: self.parameter * numpy.power(numpy.abs(x - self.point), self.power) + self.value
		self.derivative = lambda x: self.parameter * self.power * numpy.power(numpy.abs(x - self.point), self.power - 1.0) if x > self.point else - self.parameter * self.power * numpy.power(numpy.abs(x - self.point), self.power - 1.0)
		self.increasing = self.parameter > 0.0

class Exponential(Univariate):
	'''
		Exponential Univariate Function
		Mathematically, f(x) ~ p1 * exp(p2 * |x - x0|)
	'''
	def __init__(self, begin, end, value = None, point = None, shift = None, parameter1 = None, parameter2 = None, lowerlimit = None, upperlimit = None, stepsize = None, functiondepictor = None, derivativedepictor = None):
		'''
			Constructor
			: param begin : lower limit of function domain in reals
			: param end : upper limit of function domain in reals
			: param value : value of function at point
			: param point : x0, as given in its mathematical expression
			: param shift : parameter determining non stationarity of function minima
			: param parameter1 : p1, as given in its mathematical expression
			: param parameter2 : p2, as given in its mathematical expression
			: param lowerlimit : lower limit used to plot along univariate function axis
			: param upperlimit : upper limit used to plot along univariate function axis
			: param stepsize : step size used to plot along univariate function axis
			: param functiondepictor : depictor used to plot function
			: param derivativedepictor : depictor used to plot function derivative
		'''
		Univariate.__init__(self, begin, end, value, point, shift, lowerlimit, upperlimit, stepsize, functiondepictor, derivativedepictor)
		self.parameter1 = parameter1 if parameter1 is not None else 1.0
		self.parameter2 = parameter2 if parameter2 is not None else 1.0
		self.function = lambda x: self.parameter1 * numpy.exp(self.parameter2 * numpy.abs(x - self.point)) + self.value - self.parameter1
		self.derivative = lambda x: self.parameter1 * self.parameter2 * numpy.exp(self.parameter2 * (x - self.point)) if x > self.point else - self.parameter1 * self.parameter2 * numpy.exp(self.parameter2 * (x - self.point))
		self.increasing = self.parameter1 * self.parameter2 > 0.0

# piece together after applying noise, if any
class Curve(visualize.UnivariatePlot):
	'''
		Curve Univariate Piecewise Function
		Mathematically, f(x) = fi(x) if i1 <= x <= i2
	'''
	def __init__(self, functionlist, lowerlimit = None, upperlimit = None, stepsize = None, functiondepictor = None, derivativedepictor = None):
		'''
			Constructor
			: param functionlist : list of piecewise univariate functions to be composed over intervals
			: param lowerlimit : lower limit used to plot along univariate function axis
			: param upperlimit : upper limit used to plot along univariate function axis
			: param stepsize : step size used to plot along univariate function axis
			: param functiondepictor : depictor used to plot function
			: param derivativedepictor : depictor used to plot function derivative
		'''
		self.functionlist, self.rangelist = zip(*sorted(zip(functionlist, [(function.begin, function.end) for function in functionlist]), key = lambda x: x[1][0]))
		self.begin = self.rangelist[0][0]
		self.end = self.rangelist[-1][1]
		visualize.UnivariatePlot.__init__(self, lowerlimit, upperlimit, stepsize, functiondepictor, derivativedepictor)

	def function(self, point):
		'''
			Method to evaluate piecewise univariate function
			: param inputvector : point of evaluation in parameter space
			: returns : evaluated function value at point in parameter space
		'''
		for function in self.functionlist:
			if function.begin <= point <= function.end:
				return function.function(point)
		return numpy.nan

	def derivative(self, point):
		'''
			Method to evaluate piecewise univariate derivative
			: param inputvector : point of evaluation in parameter space
			: returns : evaluated derivative value at point in parameter space
		'''
		for function in self.functionlist:
			if function.begin <= point <= function.end:
				return function.derivative(point)
		return numpy.nan

	def minima(self):
		'''
			Method to calculate true mimima of univariate function
			: returns : point corresponding to minima of univariate function
		'''
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
		'''
			Method to update non stationary minima of univariate function
		'''
		for function in self.functionlist:
			function.update()
