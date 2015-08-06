import numpy

class Series:

	layers = None

	inputs = None
	outputs = None

	previousinput = None
	previousoutput = None

	def __init__(self):
		self.layers = list()

	def addlayer(self, layer):
		self.layers.append(layer)
		self.outputs = self.layers[-1].outputs
		if self.inputs is None:
			self.inputs = self.layers[-1].inputs

	def cleardeltas(self):
		for layer in self.layers:
			if layer.__class__.__name__ in ['Linear', 'Series', 'Parallel']:
				layer.cleardeltas()

	def updateweights(self):
		for layer in self.layers:
			if layer.__class__.__name__ in ['Linear', 'Series', 'Parallel']:
				layer.updateweights()

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

	def trainingsetup(self):
		for layer in self.layers:
			if layer.__class__.__name__ in ['Linear', 'Series', 'Parallel']:
				layer.trainingsetup()

	def testingsetup(self):
		for layer in self.layers:
			if layer.__class__.__name__ in ['Linear', 'Series', 'Parallel']:
				layer.trainingsetup()

	def applyvelocity(self, gamma = None):
		for layer in self.layers:
			if layer.__class__.__name__ in ['Linear', 'Series', 'Parallel']:
				layer.applyvelocity(gamma)

	def applyregularization(self, lamda = None, regularizer = None):
		for layer in self.layers:
			if layer.__class__.__name__ in ['Linear', 'Series', 'Parallel']:
				layer.applyregularization(lamda, regularizer)

	def applydropout(self, probability = None):
		for layer in self.layers:
			if layer.__class__.__name__ in ['Linear', 'Series', 'Parallel']:
				layer.applydropout(probability)

class Parallel:

	layers = None

	inputs = None
	outputs = None
	inputdimensions = None
	outputdimensions = None

	previousinput = None
	previousoutput = None

	def __init__(self):
		self.layers = list()
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

	def cleardeltas(self):
		for layer in self.layers:
			layer.cleardeltas()

	def updateweights(self):
		for layer in self.layers:
			layer.updateweights()

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

	def trainingsetup(self):
		for layer in self.layers:
			if layer.__class__.__name__ in ['Linear', 'Series', 'Parallel']:
				layer.trainingsetup()

	def testingsetup(self):
		for layer in self.layers:
			if layer.__class__.__name__ in ['Linear', 'Series', 'Parallel']:
				layer.trainingsetup()

	def applyvelocity(self, gamma = None):
		for layer in self.layers:
			if layer.__class__.__name__ in ['Linear', 'Series', 'Parallel']:
				layer.applyvelocity(gamma)

	def applyregularization(self, lamda = None, regularizer = None):
		for layer in self.layers:
			if layer.__class__.__name__ in ['Linear', 'Series', 'Parallel']:
				layer.applyregularization(lamda, regularizer)

	def applydropout(self, probability = None):
		for layer in self.layers:
			if layer.__class__.__name__ in ['Linear', 'Series', 'Parallel']:
				layer.applydropout(probability)
