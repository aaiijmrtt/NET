import numpy

class Container:

	listoflayers = ['Linear', 'Normalizer', 'Convolutional', 'Perceptron', 'AutoEncoder', 'HopfieldNetwork', 'GaussianRB', 'MultiQuadraticRB', 'InverseMultiQuadraticRB', 'ThinPlateSplineRB', 'CubicRB', 'LinearRB', 'RestrictedBoltzmann', 'ManhattanSO', 'EuclideanSquaredSO']
	listofcontainers = ['Series', 'Parallel', 'Recurrent']

	def __init__(self):
		self.layers = list()
		self.inputs = None
		self.outputs = None
		self.previousinput = None
		self.previousoutput = None

	def accumulate(self, inputvector):
		for layer in self.layers:
			if layer.__class__.__name__ in ['Normalizer'] + Container.listofcontainers:
				layer.accumulate(inputvector)

	def cleardeltas(self):
		for layer in self.layers:
			if layer.__class__.__name__ in Container.listoflayers + Container.listofcontainers:
				layer.cleardeltas()

	def updateweights(self):
		for layer in self.layers:
			if layer.__class__.__name__ in Container.listoflayers + Container.listofcontainers:
				layer.updateweights()

	def normalize(self):
		for layer in self.layers:
			if layer.__class__.__name__ in ['Normalizer'] + Container.listofcontainers:
				layer.normalize()

	def accumulatingsetup(self):
		for layer in self.layers:
			if layer.__class__.__name__ in ['Normalizer'] + Container.listofcontainers:
				layer.accumulatingsetup()

	def timingsetup(self):
		for layer in self.layers:
			if layer.__class__.__name__ in Container.listofcontainers:
				layer.timingsetup()

	def trainingsetup(self):
		for layer in self.layers:
			if layer.__class__.__name__ in Container.listoflayers + Container.listofcontainers:
				layer.trainingsetup()

	def testingsetup(self):
		for layer in self.layers:
			if layer.__class__.__name__ in Container.listoflayers + Container.listofcontainers:
				layer.trainingsetup()

	def applylearningrate(self, alpha = None):
		for layer in self.layers:
			if layer.__class__.__name__ in Container.listoflayers + Container.listofcontainers:
				layer.applylearningrate(alpha)

	def applydecayrate(self, eta = None):
		for layer in self.layers:
			if layer.__class__.__name__ in Container.listoflayers + Container.listofcontainers:
				layer.applydecayrate(eta)

	def applyvelocity(self, gamma = None):
		for layer in self.layers:
			if layer.__class__.__name__ in Container.listoflayers + Container.listofcontainers:
				layer.applyvelocity(gamma)

	def applyregularization(self, lamda = None, regularizer = None):
		for layer in self.layers:
			if layer.__class__.__name__ in Container.listoflayers + Container.listofcontainers:
				layer.applyregularization(lamda, regularizer)

	def applydropout(self, probability = None):
		for layer in self.layers:
			if layer.__class__.__name__ in Container.listoflayers + Container.listofcontainers:
				layer.applydropout(probability)

	def applyadaptivegain(self, tau = None, maximum = None, minimum = None):
		for layer in self.layers:
			if layer.__class__.__name__ in Container.listoflayers + Container.listofcontainers:
				layer.applyadaptivegain(tau, maximum, minimum)

	def applyrootmeansquarepropagation(self, meu = None):
		for layer in self.layers:
			if layer.__class__.__name__ in Container.listoflayers + Container.listofcontainers:
				layer.applyrootmeansquarepropagation(meu)

	def applyadaptivegradient(self):
		for layer in self.layers:
			if layer.__class__.__name__ in Container.listoflayers + Container.listofcontainers:
				layer.applyadaptivegradient()

class Series(Container):

	def __init__(self):
		Container.__init__(self)

	def addlayer(self, layer):
		self.layers.append(layer)
		self.outputs = self.layers[-1].outputs
		if self.inputs is None:
			self.inputs = self.layers[-1].inputs

	def feedforward(self, inputvector):
		self.previousinput = inputvector
		self.previousoutput = self.previousinput
		for layer in self.layers:
			self.previousoutput = layer.feedforward(self.previousoutput)
		return self.previousoutput

	def backpropagate(self, outputvector):
		for layer in reversed(self.layers):
			outputvector = layer.backpropagate(outputvector)
		return outputvector

class Parallel(Container):

	def __init__(self):
		Container.__init__(self)
		self.inputdimensions = [0]
		self.outputdimensions = [0]
		self.inputs = 0
		self.outputs = 0

	def addlayer(self, layer):
		self.layers.append(layer)
		self.inputs += self.layers[-1].inputs
		self.outputs += self.layers[-1].outputs
		self.inputdimensions.append(self.inputdimensions[-1] + self.layers[-1].inputs)
		self.outputdimensions.append(self.outputdimensions[-1] + self.layers[-1].outputs)

	def feedforward(self, inputvector):
		self.previousinput = inputvector
		layeroutputs = list()
		for layer, layerinput in zip(self.layers, numpy.split(self.previousinput, self.inputdimensions[1: -1])):
			layeroutputs.append(layer.feedforward(layerinput))
		self.previousoutput = numpy.concatenate(layeroutputs)
		return self.previousoutput

	def backpropagate(self, outputvector):
		layerdeltas = list()
		for layer, layeroutput in zip(self.layers, numpy.split(outputvector, self.outputdimensions[1: -1])):
			layerdeltas.append(layer.backpropagate(layeroutput))
		return numpy.concatenate(layerdeltas)

class Recurrent(Container):

	def __init__(self, hiddens, layer):
		Container.__init__(self)
		self.layers.append(layer)
		self.hiddens = hiddens
		self.inputs = self.layers[0].inputs - self.hiddens
		self.outputs = self.layers[0].outputs - self.hiddens
		self.previoushiddens = None
		self.previousdeltas = None

	def feedforward(self, inputvector):
		self.previousinput = inputvector
		outputvector = self.layers[0].feedforward(numpy.concatenate([self.previoushiddens, self.previousinput]))
		self.previoushiddens, self.previousoutput = numpy.split(outputvector, [self.hiddens])
		return self.previousoutput

	def backpropagate(self, outputvector):
		deltavector = self.layers[0].backpropagate(numpy.concatenate([numpy.zeros((self.hiddens + self.outputs, 1), dtype = float)]))
		deltahidden, deltainput = numpy.split(deltavector, [self.hiddens])
		return deltainput

	def timingsetup(self):
		self.previoushiddens = numpy.zeros((self.hiddens, 1), dtype = float)
		Container.timingsetup(self)
