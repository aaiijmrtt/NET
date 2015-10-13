'''
	Module containing Hopfiled Network Layers.
	Classes embody Parametric Synchronous Symmetric Recurrent Layers,
	used to recall stored patterns.
'''
import numpy
from . import layer, error

class HopfieldNetwork(layer.Layer):
	'''
		Hopfield Network Layer
		Mathematically, f(x) = W * x where W(i)(j) = W(j)(i)
										and W(i)(i) = 0
	'''
	def __init__(self, inputs, alpha = None):
		'''
			Constructor
			: param inputs : dimension of input (and output) feature space
			: param alpha : learning rate constant hyperparameter
		'''
		layer.Layer.__init__(self, inputs, inputs, alpha)
		self.parameters['weights'] = numpy.random.normal(0.0, 1.0 / numpy.sqrt(self.inputs), (self.inputs, self.inputs))
		self.parameters['weights'] = numpy.add(self.parameters['weights'], numpy.transpose(self.parameters['weights']))
		self.cleardeltas()

	def cleardeltas(self):
		'''
			Method to clear accumulated parameter changes
		'''
		self.deltaparameters = self.modifier.cleardeltas()
		for i in range(self.inputs):
			self.parameters['weights'][i][i] = 0.0

	def updateweights(self):
		'''
			Method to update weights based on accumulated parameter changes
		'''
		self.deltaparameters['weights'] = numpy.add(self.deltaparameters['weights'], numpy.transpose(self.deltaparameters['weights']))
		self.deltaparameters = self.modifier.updateweights()
		for parameter in self.parameters:
			self.parameters[parameter] = numpy.subtract(self.parameters[parameter], self.deltaparameters[parameter])
		self.cleardeltas()

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		self.previousinput = self.modifier.feedforward(inputvector)
		self.previousoutput = numpy.dot(self.parameters['weights'], self.previousinput)
		return self.previousoutput

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		self.deltaparameters['weights'] = numpy.add(self.deltaparameters['weights'], numpy.dot(outputvector, numpy.transpose(self.previousinput)))
		return numpy.dot(numpy.transpose(self.parameters['weights']), outputvector)

	def pretrain(self, trainingset, batch = 1, iterations = 1, criterion = None):
		'''
			Method to pretrain parameters using Hebbian Learning
			: param trainingset : unsupervised training set
			: param batch : training minibatch size
			: param iterations : iteration threshold for termination
			: param criterion : criterion used to quantify reconstruction error
			: returns : elementwise reconstruction error on termination
		'''
		self.parameters['weights'] = numpy.zeros((self.inputs, self.inputs), dtype = float)
		if criterion is None:
			criterion = error.MeanSquared(self.inputs)
		self.trainingsetup()
		for i in range(iterations):
			for j in range(len(trainingset)):
				if j % batch == 0:
					self.updateweights()
				self.feedforward(trainingset[j])
				self.backpropagate(-trainingset[j])
		self.testingsetup()
		errorvector = numpy.zeros((self.outputs, 1), dtype = float)
		for vector in trainingset:
			errorvector = numpy.add(errorvector, criterion.compute(self.feedforward(vector), vector))
		errorvector = numpy.divide(errorvector, len(trainingset))
		return errorvector
