import numpy
from . import layer, error

class RadialBasis(layer.Layer):

	def __init__(self, inputs, outputs, alpha = None):
		layer.Layer.__init__(self, inputs, outputs, alpha)
		self.parameters = dict()
		self.deltaparameters = dict()
		self.parameters['weights'] = numpy.random.normal(0.0, 1.0 / numpy.sqrt(self.inputs), (self.outputs, self.inputs))
		self.parameters['biases'] = numpy.random.normal(0.0, 1.0 / numpy.sqrt(self.inputs), (self.outputs, 1))
		self.function = None
		self.functionderivative = None
		self.weightsderivative = None
		self.biasesderivative = None
		self.cleardeltas()

	def feedforward(self, inputvector):
		self.previousinput = inputvector
		self.previousradius = numpy.empty((self.outputs, 1), dtype = float)
		self.previousoutput = numpy.empty((self.outputs, 1), dtype = float)
		for i in range(self.outputs):
			centrevector = self.parameters['weights'][i].reshape((self.inputs, 1))
			self.previousradius[i][0] = numpy.sum(numpy.square(numpy.subtract(self.previousinput, centrevector))) ** 0.5
			self.previousoutput[i][0] = self.function(self.previousradius[i][0], self.parameters['biases'][i][0])
		return self.previousoutput

	def backpropagate(self, outputvector):
		deltainputs = numpy.zeros((self.inputs, 1), dtype = float)
		for i in range(self.outputs):
			centrevector = self.parameters['weights'][i].reshape((self.inputs, 1))
			deltainputs = numpy.add(deltainputs, numpy.multiply(outputvector[i][0], self.functionderivative(self.previousradius[i][0], self.previousinput, centrevector, self.parameters['biases'][i][0], self.previousoutput[i][0])))
			self.deltaparameters['weights'][i] = numpy.add(self.deltaparameters['weights'][i], numpy.multiply(outputvector[i][0], numpy.transpose(self.weightsderivative(self.previousradius[i][0], self.previousinput, centrevector, self.parameters['biases'][i][0], self.previousoutput[i][0]))))
			self.deltaparameters['biases'][i][0] += outputvector[i][0] * self.biasesderivative(self.previousradius[i][0], self.previousinput, centrevector, self.parameters['biases'][i][0], self.previousoutput[i][0])
		return deltainputs

	def pretrain(self, trainingset, threshold = 0.0001, iterations = 10, criterion = None): # K Means Clustering
		if criterion is None:
			criterion = error.MeanSquared(self.inputs)
		for iterations in range(iterations):
			clusters = [[self.parameters['weights'][i].reshape((self.inputs, 1))] for i in range(self.outputs)]
			for point in trainingset:
				bestdistance = float('inf')
				bestindex = -1
				for i in range(len(clusters)):
					distance = numpy.sum(criterion.compute(point, clusters[i][0]))
					if distance < bestdistance:
						bestdistance = distance
						bestindex = i
				clusters[bestindex].append(point)
			for cluster in clusters:
				if len(cluster) > 1:
					cluster[0] = numpy.mean(cluster[1:], axis = 0)
			maximumdistance = float('-inf')
			for cluster in clusters:
				if len(cluster) > 1:
					for point in cluster[1: ]:
						distance = numpy.sum(criterion.compute(point, cluster[0]))
						if distance > maximumdistance:
							maximumdistance = distance
			if maximumdistance < threshold:
				break
		for i in range(len(self.parameters['weights'])):
			self.parameters['weights'][i] = numpy.transpose(clusters[i][0])
		return maximumdistance

class GaussianRB(RadialBasis):

	def __init__(self, inputs, outputs, alpha = None):
		RadialBasis.__init__(self, inputs, outputs, alpha)
		self.function = lambda inputradius, coefficient: numpy.exp(-inputradius / coefficient ** 2)
		self.functionderivative = lambda inputradius, inputvector, centrevector, coefficient, output: numpy.multiply(2.0 * output / (coefficient ** 2), numpy.subtract(centrevector, inputvector))
		self.weightsderivative = lambda inputradius, inputvector, centrevector, coefficient, output: numpy.multiply(2.0 * output / (coefficient ** 2), numpy.subtract(inputvector, centrevector))
		self.biasesderivative = lambda inputradius, inputvector, centrevector, coefficient, output: 2.0 * output * numpy.sum(numpy.square(numpy.subtract(inputvector, centrevector))) / (coefficient ** 3)

class MultiQuadraticRB(RadialBasis):

	def __init__(self, inputs, outputs, beta = None, alpha = None):
		RadialBasis.__init__(self, inputs, outputs, alpha)
		self.beta = beta if beta is not None else 0.5
		self.function = lambda inputradius, coefficient: (inputradius ** 2 + coefficient ** 2) ** self.beta
		self.functionderivative = lambda inputradius, inputvector, centrevector, coefficient, output: numpy.multiply(self.beta * (inputradius ** 2 + coefficient ** 2) ** (self.beta - 1.0) * 2.0, numpy.subtract(inputvector, centrevector))
		self.weightsderivative = lambda inputradius, inputvector, centrevector, coefficient, output: numpy.multiply(self.beta * (inputradius ** 2 + coefficient ** 2) ** (self.beta - 1.0) * 2.0, numpy.subtract(centrevector, inputvector))
		self.biasesderivative = lambda inputradius, inputvector, centrevector, coefficient, output: self.beta * (inputradius ** 2 + coefficient ** 2) ** (self.beta - 1.0) * 2.0 * coefficient

class InverseMultiQuadraticRB(RadialBasis):

	def __init__(self, inputs, outputs, beta = None, alpha = None):
		RadialBasis.__init__(self, inputs, outputs, alpha)
		self.beta = beta if beta is not None else 0.5
		self.function = lambda inputradius, coefficient: (inputradius ** 2 + coefficient ** 2) ** (-self.beta)
		self.functionderivative = lambda inputradius, inputvector, centrevector, coefficient, output: numpy.multiply(-self.beta * (inputradius ** 2 + coefficient ** 2) ** (-self.beta - 1.0) * 2.0, numpy.subtract(inputvector, centrevector))
		self.weightsderivative = lambda inputradius, inputvector, centrevector, coefficient, output: numpy.multiply(-self.beta * (inputradius ** 2 + coefficient ** 2) ** (-self.beta - 1.0) * 2.0, numpy.subtract(centrevector, inputvector))
		self.biasesderivative = lambda inputradius, inputvector, centrevector, coefficient, output: -self.beta * (inputradius ** 2 + coefficient ** 2) ** (-self.beta - 1.0) * 2.0 * coefficient

class ThinPlateSplineRB(RadialBasis):

	def __init__(self, inputs, outputs, alpha = None):
		RadialBasis.__init__(self, inputs, outputs, alpha)
		self.function = lambda inputradius, coefficient: inputradius ** 2 * numpy.log(inputradius)
		self.functionderivative = lambda inputradius, inputvector, centrevector, coefficient, output: numpy.multiply((2.0 * numpy.log(inputradius) + 1.0), numpy.subtract(inputvector, centrevector))
		self.weightsderivative = lambda inputradius, inputvector, centrevector, coefficient, output: numpy.multiply((2.0 * numpy.log(inputradius) + 1.0), numpy.subtract(centrevector, inputvector))
		self.biasesderivative = lambda inputradius, inputvector, centrevector, coefficient, output: 0.0

class CubicRB(RadialBasis):

	def __init__(self, inputs, outputs, alpha = None):
		RadialBasis.__init__(self, inputs, outputs, alpha)
		self.function = lambda inputradius, coefficient: inputradius ** 3
		self.functionderivative = lambda inputradius, inputvector, centrevector, coefficient, output: numpy.multiply(3.0 * inputradius, numpy.subtract(inputvector, centrevector))
		self.weightsderivative = lambda inputradius, inputvector, centrevector, coefficient, output: numpy.multiply(3.0 * inputradius, numpy.subtract(centrevector, inputvector))
		self.biasesderivative = lambda inputradius, inputvector, centrevector, coefficient, output: 0.0

class LinearRB(RadialBasis):

	def __init__(self, inputs, outputs, alpha = None):
		RadialBasis.__init__(self, inputs, outputs, alpha)
		self.function = lambda inputradius, coefficient: inputradius
		self.functionderivative = lambda inputradius, inputvector, centrevector, coefficient, output: numpy.divide(numpy.subtract(inputvector, centrevector), output)
		self.weightsderivative = lambda inputradius, inputvector, centrevector, coefficient, output: numpy.divide(numpy.subtract(centrevector, inputvector), output)
		self.biasesderivative = lambda inputradius, inputvector, centrevector, coefficient, output: 0.0
