'''
	Module containing Visualizers.
	Classes embody Function Plots,
	for visualization of data.
'''
import numpy, matplotlib.pyplot

class UnivariatePlot:
	'''
		Univariate Function Plots
	'''
	# assumed to have a univariate function already defined
	def __init__(self, lowerlimit = None, upperlimit = None, stepsize = None, functiondepictor = None, derivativedepictor = None):
		'''
			Constructor
			: param lowerlimit : lower limit used to plot along univariate function axis
			: param upperlimit : upper limit used to plot along univariate function axis
			: param stepsize : step size used to plot along univariate function axis
			: param functiondepictor : depictor used to plot function
			: param derivativedepictor : depictor used to plot function derivative
		'''
		self.lowerlimit = lowerlimit if lowerlimit is not None else self.begin
		self.upperlimit = upperlimit if lowerlimit is not None else self.end
		self.stepsize = stepsize if stepsize is not None else 0.1
		self.functiondepictor = functiondepictor if functiondepictor is not None else 'b^'
		self.derivativedepictor = derivativedepictor if derivativedepictor is not None else 'r^'

	def plot(self):
		'''
			Method to plot function and derivative values of univariate function
		'''
		matplotlib.pyplot.figure('Univariate Plot')
		points = numpy.arange(self.lowerlimit, self.upperlimit, self.stepsize)
		matplotlib.pyplot.plot(points, [self.function(point) for point in points], self.functiondepictor)
		matplotlib.pyplot.plot(points, [self.derivative(point) for point in points], self.derivativedepictor)
		matplotlib.pyplot.show()

class MultivariatePlot:
	'''
		Multivariate Function Plots
	'''
	# assumed to have a multivariate function already defined
	def __init__(self, lowerlimits = None, upperlimits = None, stepsizes = None, functiondepictor = None, derivativedepictor = None):
		'''
			Constructor
			: param lowerlimits : lower limits used to plot along univariate function axes
			: param upperlimits : upper limits used to plot along univariate function axes
			: param stepsizes : step sizes used to plot along univariate function axes
			: param functiondepictor : depictor used to plot function
			: param derivativedepictor : depictor used to plot function derivative
		'''
		self.lowerlimits = lowerlimits if lowerlimits is not None else [univariate.begin for univariate in self.univariatelist]
		self.upperlimits = upperlimits if lowerlimits is not None else [univariate.end for univariate in self.univariatelist]
		self.stepsizes = stepsizes if stepsizes is not None else [univariate.stepsize for univariate in self.univariatelist]
		self.functiondepictor = functiondepictor if functiondepictor is not None else 'b^'
		self.derivativedepictor = derivativedepictor if derivativedepictor is not None else 'r^'

	def plot(self, point):
		'''
			Method to plot partial function and partial derivative values of multivariate function about a point
			: param point : point of evaluation in parameter space
		'''
		matplotlib.pyplot.figure('Multivariate Plot')
		for i in range(len(self.univariatelist)):
			matplotlib.pyplot.subplot(len(self.univariatelist), 1, i)
			points = numpy.arange(self.lowerlimits[i], self.upperlimits[i], self.stepsizes[i])
			deltas = list()
			for delta in points:
				deltas.append(point.copy())
				deltas[-1][i] = delta
			matplotlib.pyplot.plot(points, [self.function(delta) for delta in deltas], self.functiondepictor)
			matplotlib.pyplot.plot(points, [self.derivative(delta)[i] for delta in deltas], self.derivativedepictor)
		matplotlib.pyplot.show()

	def contourplot(self, point, dimensionx, dimensiony):
		'''
			Method to plot function values of a multivariate function
			: param point : point of evaluation in parameter space
			: param dimensionx : dimension to plot on x axis
			: param dimensiony : dimension to plot on y axis
		'''
		matplotlib.pyplot.figure('Multivariate Contour Plot')
		pointsx = numpy.arange(self.lowerlimits[dimensionx], self.upperlimits[dimensionx], self.stepsizes[dimensionx])
		pointsy = numpy.arange(self.lowerlimits[dimensiony], self.upperlimits[dimensiony], self.stepsizes[dimensiony])
		values = numpy.empty((len(pointsy), len(pointsx)), dtype = float)
		for i in range(len(pointsx)):
			for j in range(len(pointsy)):
				z = point.copy()
				z[dimensionx][0] = pointsx[i]
				z[dimensiony][0] = pointsy[j]
				values[j][i] = self.function(z)
		CS = matplotlib.pyplot.contour(pointsx, pointsy, values)
		matplotlib.pyplot.clabel(CS, fontsize = 10)
		matplotlib.pyplot.show()

class HintonPlot:
	'''
		Hinton Matrix Plots
	'''
	def plot(self, matrix, size = None):
		'''
			Method to plot Hinton Diagram of matrix
			: param matrix : matrix to be plotted
			: param size : scaling of blocks in plot
		'''
		matplotlib.pyplot.figure('Hinton Plot')
		matplotlib.pyplot.gca().set_axis_bgcolor('gray')
		size = size if size is not None else 5000 / max(matrix.shape)
		columns = list()
		rows = list()
		values = list()
		sizes = list()
		maximum = numpy.amax(numpy.abs(matrix))
		for (row, column), value in numpy.ndenumerate(matrix):
			columns.append(column)
			rows.append(row)
			if value > 0.0:
				values.append('white')
			else:
				values.append('black')
			sizes.append(size * numpy.abs(value) / maximum)
		matplotlib.pyplot.scatter(columns, rows, s = sizes, c = values, marker = 's')
		matplotlib.pyplot.show()
