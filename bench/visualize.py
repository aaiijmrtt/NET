import numpy, matplotlib.pyplot, copy

class UnivariatePlot:

	def plot(self, lowerlimit = None, upperlimit = None, stepsize = None, functiondepictor = None, derivativedepictor = None):
		matplotlib.pyplot.figure(0)
		lowerlimit = lowerlimit if lowerlimit is not None else self.begin
		upperlimit = upperlimit if upperlimit is not None else self.end
		stepsize = stepsize if stepsize is not None else 0.1
		functiondepictor = functiondepictor if functiondepictor is not None else 'b^'
		derivativedepictor = derivativedepictor if derivativedepictor is not None else 'r^'
		points = numpy.arange(lowerlimit, upperlimit, stepsize)
		matplotlib.pyplot.plot(points, [self.function(point) for point in points], functiondepictor)
		matplotlib.pyplot.plot(points, [self.derivative(point) for point in points], derivativedepictor)
		matplotlib.pyplot.show()

class MultivariatePlot:

	def plot(self, point, lowerlimit = None, upperlimit = None, stepsize = None, functiondepictor = None, derivativedepictor = None):
		matplotlib.pyplot.figure(0)
		for i in range(len(self.univariatelist)):
			matplotlib.pyplot.subplot(len(self.univariatelist), 1, i)
			lowerlimit = lowerlimit if lowerlimit is not None else self.univariatelist[i].begin
			upperlimit = upperlimit if upperlimit is not None else self.univariatelist[i].end
			stepsize = stepsize if stepsize is not None else 0.1
			functiondepictor = functiondepictor if functiondepictor is not None else 'b^'
			derivativedepictor = derivativedepictor if derivativedepictor is not None else 'r^'
			points = numpy.arange(lowerlimit, upperlimit, stepsize)
			deltas = list()
			for delta in points:
				deltas.append(copy.deepcopy(point))
				deltas[-1][i] = delta
			matplotlib.pyplot.plot(points, [self.function(delta) for delta in deltas], functiondepictor)
			matplotlib.pyplot.plot(points, [self.derivative(delta)[i] for delta in deltas], derivativedepictor)
		matplotlib.pyplot.show()

class HintonPlot:

	def plot(self, matrix, size = None):
		matplotlib.pyplot.figure(0)
		size = size if size is not None else 50
		columns = list()
		rows = list()
		values = list()
		for (row, column), value in numpy.ndenumerate(matrix):
			columns.append(column)
			rows.append(row)
			values.append(value)
		matplotlib.pyplot.scatter(columns, rows, s = size, c = values)
		matplotlib.pyplot.show()
