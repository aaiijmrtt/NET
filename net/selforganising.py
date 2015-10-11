import numpy
from . import layer, error

class SelfOrganising(layer.Layer):

	exponentialneighbourhood = numpy.vectorize(lambda x, y: numpy.exp(x - y))
	inverseneighbourhood = numpy.vectorize(lambda x, y: 1.0 / (1.0 + y - x))
	knockerneighbourhood = numpy.vectorize(lambda x, y: 1.0 if x == y else 0.0)

	def __init__(self, inputs, outputs, alpha = None):
		layer.Layer.__init__(self, inputs, outputs, alpha)
		self.parameters = dict()
		self.deltaparameters = dict()
		self.parameters['weights'] = numpy.random.normal(0.0, 1.0 / numpy.sqrt(self.inputs), (self.outputs, self.inputs))
		self.function = None
		self.functionderivative = None
		self.weightsderivative = None
		self.cleardeltas()

	def feedforward(self, inputvector):
		self.previousinput = inputvector
		self.previousradius = numpy.empty((self.outputs, 1), dtype = float)
		self.previousoutput = numpy.zeros((self.outputs, 1), dtype = float)
		for i in range(self.outputs):
			self.previousradius[i][0] = self.function(self.previousinput, self.parameters['weights'][i].reshape((self.inputs, 1)))
		self.previousoutput[numpy.argmin(self.previousradius)][0] = 1.0
		return self.previousoutput

	def backpropagate(self, outputvector):
		index = numpy.argmin(self.previousradius)
		centrevector = self.parameters['weights'][index].reshape((self.inputs, 1))
		self.deltaparameters['weights'][index] = numpy.add(self.deltaparameters['weights'][index], numpy.multiply(outputvector[index][0], numpy.transpose(self.weightsderivative(self.previousradius[index][0], self.previousinput, centrevector))))
		return numpy.multiply(outputvector[index][0], self.functionderivative(self.previousradius[index][0], self.previousinput, centrevector))

	def pretrain(self, trainingset, threshold = 0.0001, batch = 1, iterations = 10, neighbourhood = None): # Competitive Learning
		if neighbourhood is None:
			neighbourhood = SelfOrganising.exponentialneighbourhood
		self.trainingsetup()
		for i in range(iterations):
			for j in range(len(trainingset)):
				if j % batch == 0:
					self.updateweights()
				self.feedforward(trainingset[j])
				closest = self.previousradius[numpy.argmin(self.previousoutput)][0]
				for k in range(self.outputs):
					factor = neighbourhood(closest, self.previousradius[k][0])
					centrevector = self.parameters['weights'][k].reshape((self.inputs, 1))
					self.deltaparameters['weights'][k] = numpy.add(self.deltaparameters['weights'][k], numpy.multiply(factor, numpy.transpose(numpy.subtract(centrevector, trainingset[j]))))
			maximumdistance = float('-inf')
			for vector in trainingset:
				self.feedforward(vector)
				closest = self.previousradius[numpy.argmin(self.previousoutput)][0]
				if closest > maximumdistance:
					maximumdistance = closest
			if maximumdistance < threshold:
				break
		return maximumdistance

class ManhattanSO(SelfOrganising):

	def __init__(self, inputs, outputs, alpha = None):
		SelfOrganising.__init__(self, inputs, outputs, alpha)
		self.function = lambda inputvector, centrevector: numpy.sum(numpy.abs(numpy.subtract(inputvector, centrevector)))
		self.functionderivative = lambda inputradius, inputvector, centrevector: numpy.sign(numpy.subtract(inputvector, centrevector))
		self.weightsderivative = lambda inputradius, inputvector, centrevector: numpy.sign(numpy.subtract(centrevector, inputvector))

class EuclideanSquaredSO(SelfOrganising):

	def __init__(self, inputs, outputs, alpha = None):
		SelfOrganising.__init__(self, inputs, outputs, alpha)
		self.function = lambda inputvector, centrevector: numpy.sqrt(numpy.sum(numpy.square(numpy.subtract(inputvector, centrevector))))
		self.functionderivative = lambda inputradius, inputvector, centrevector: numpy.divide(numpy.subtract(inputvector, centrevector), inputradius)
		self.weightsderivative = lambda inputradius, inputvector, centrevector: numpy.divide(numpy.subtract(centrevector, inputvector), inputradius)
