import numpy
from . import modifier

class Layer:

	def __init__(self, inputs, outputs, alpha = None):
		self.inputs = inputs
		self.outputs = outputs
		self.modifier = modifier.Modifier(self)
		self.applylearningrate(alpha)
		self.parameters = dict()
		self.deltaparameters = dict()
		self.previousinput = None
		self.previousoutput = None

	def cleardeltas(self):
		self.deltaparameters = self.modifier.cleardeltas()

	def updateweights(self):
		self.deltaparameters = self.modifier.updateweights()
		for parameter in self.parameters:
			self.parameters[parameter] = numpy.subtract(self.parameters[parameter], self.deltaparameters[parameter])
		self.cleardeltas()

	def trainingsetup(self):
		self.cleardeltas()
		self.modifier.trainingsetup()

	def testingsetup(self):
		self.modifier.testingsetup()

	def applylearningrate(self, alpha = None):
		self.modifier.applylearningrate(alpha)

	def applydecayrate(self, eta = None):
		self.modifier.applydecayrate(eta)

	def applyvelocity(self, gamma = None):
		self.modifier.applyvelocity(gamma)

	def applyregularization(self, lamda = None, regularizer = None):
		self.modifier.applyregularization(lamda, regularizer)

	def applydropout(self, rho = None):
		self.modifier.applydropout(rho)

	def applyadaptivegain(self, tau = None, maximum = None, minimum = None):
		self.modifier.applyadaptivegain(tau, maximum, minimum)

	def applyrootmeansquarepropagation(self, meu = None):
		self.modifier.applyrootmeansquarepropagation(meu)

	def applyadaptivegradient(self):
		self.modifier.applyadaptivegradient()

class Linear(Layer):

	def __init__(self, inputs, outputs, alpha = None):
		Layer.__init__(self, inputs, outputs, alpha)
		self.parameters['weights'] = numpy.random.normal(0.0, 1.0 / numpy.sqrt(self.inputs), (self.outputs, self.inputs))
		self.parameters['biases'] = numpy.random.normal(0.0, 1.0 / numpy.sqrt(self.inputs), (self.outputs, 1))
		self.cleardeltas()

	def feedforward(self, inputvector):
		self.previousinput = self.modifier.feedforward(inputvector)
		self.previousoutput = numpy.add(numpy.dot(self.parameters['weights'], self.previousinput), self.parameters['biases'])
		return self.previousoutput

	def backpropagate(self, outputvector):
		self.deltaparameters['weights'] = numpy.add(self.deltaparameters['weights'], numpy.dot(outputvector, numpy.transpose(self.previousinput)))
		self.deltaparameters['biases'] = numpy.add(self.deltaparameters['biases'], outputvector)
		return numpy.dot(numpy.transpose(self.parameters['weights']), outputvector)

class Normalizer(Layer):

	epsilon = 0.0001

	def __init__(self, inputs, alpha = None):
		Layer.__init__(self, inputs, inputs, alpha)
		self.parameters['weights'] = numpy.ones((self.inputs, 1), dtype = float)
		self.parameters['biases'] = numpy.zeros((self.inputs, 1), dtype = float)
		self.mean = numpy.zeros((self.inputs, 1), dtype = float)
		self.variance = numpy.ones((self.inputs, 1), dtype = float)
		self.batch = 1
		self.cleardeltas()
		self.linearsum = None
		self.quadraticsum = None

	def accumulate(self, inputvector):
		self.linearsum = numpy.add(self.linearsum, inputvector)
		self.quadraticsum = numpy.add(self.quadraticsum, numpy.square(inputvector))
		self.batch += 1

	def feedforward(self, inputvector): # ignores dropout
		self.previousinput = inputvector
		self.previousnormalized = numpy.divide(numpy.subtract(self.previousinput, self.mean), numpy.sqrt(numpy.add(Normalizer.epsilon, self.variance)))
		self.previousoutput = numpy.add(numpy.multiply(self.parameters['weights'], self.previousnormalized), self.parameters['biases'])
		return self.previousoutput

	def backpropagate(self, outputvector): # ignores dropout
		self.deltaparameters['weights'] = numpy.add(self.deltaparameters['weights'], numpy.multiply(outputvector, self.previousnormalized))
		self.deltaparameters['biases'] = numpy.add(self.deltaparameters['biases'], outputvector)
		return numpy.multiply(numpy.divide(self.parameters['weights'], self.batch), numpy.divide(numpy.subtract(self.batch - 1, numpy.square(self.previousnormalized)), numpy.sqrt(numpy.add(Normalizer.epsilon, self.variance))))

	def normalize(self):
		self.mean = numpy.divide(self.linearsum, self.batch)
		self.variance = numpy.subtract(numpy.divide(self.quadraticsum, self.batch), numpy.square(self.mean))

	def accumulatingsetup(self):
		self.batch = 0
		self.linearsum = numpy.zeros((self.inputs, 1), dtype = float)
		self.quadraticsum = numpy.zeros((self.inputs, 1), dtype = float)
