'''
	Module containing Long Short Term Memory Layers.
	Classes embody Parametric Recurrent NonLinear Layers,
	used to construct a neural network.
'''
import math, numpy
from . import configure, connector, layer, transfer

class LongShortTermMemory(layer.Layer):
	'''
		Base Class for all Long Short Term Memory Layers
		Mathematically, f(x(t)) = og(x(t)) * o(x(t))
						o(x(t)) = sigma3(h(t))
						h(x(t)) = ig(x(t)) * i(x(t)) + fg(x(t)) * h(t-1)
	'''
	def __init__(self, inputs, outputs, alpha = None):
		'''
			Constructor
			: param inputs : dimension of input feature space
			: param outputs : dimension of output feature space
			: param alpha : learning rate constant hyperparameter
		'''
		layer.Layer.__init__(self, inputs, outputs, alpha)
		self.units['inputgate'] = None
		self.units['forgetgate'] = None
		self.units['outputgate'] = None
		self.units['inputactivator'] = None
		self.units['outputactivator'] = None
		self.history['inputgate'] = None
		self.history['outputgate'] = None
		self.history['forgetgate'] = None
		self.history['inputactivator'] = None
		self.history['outputactivator'] = None
		self.history['hidden'] = None
		self.history['deltahidden'] = None

	def recur(self, inputvector, gatevector):
		'''
			Method to recur a vector through the layer
			: param inputvector : vector in input feature space
			: param gatevector : vector in gate input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		self.history['input'].append(inputvector)
		self.history['inputgate'].append(self.units['inputgate'].feedforward(gatevector))
		self.history['forgetgate'].append(self.units['forgetgate'].feedforward(gatevector))
		self.history['outputgate'].append(self.units['outputgate'].feedforward(gatevector))
		self.history['inputactivator'].append(self.units['inputactivator'].feedforward(self.history['input'][-1]))
		self.history['hidden'].append(configure.functions['add'](configure.functions['multiply'](self.history['inputgate'][-1], self.history['inputactivator'][-1]), configure.functions['multiply'](self.history['forgetgate'][-1], self.history['hidden'][-1])))
		self.history['outputactivator'].append(self.units['outputactivator'].feedforward(self.history['hidden'][-1]))
		self.history['output'].append(configure.functions['multiply'](self.history['outputgate'][-1], self.history['outputactivator'][-1]))
		return self.history['output'][-1]

	def unrecur(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		self.history['output'].pop()
		self.history['hidden'].pop()
		self.history['input'].pop()
		self.units['outputgate'].backpropagate(configure.functions['multiply'](self.history['outputactivator'].pop(), outputvector))
		outputvector = configure.functions['add'](self.units['outputactivator'].backpropagate(configure.functions['multiply'](self.history['outputgate'].pop(), outputvector)), self.history['deltahidden'][-1])
		self.history['deltahidden'].append(configure.functions['multiply'](self.history['forgetgate'].pop(), outputvector))
		self.units['forgetgate'].backpropagate(configure.functions['multiply'](self.history['hidden'][-1], outputvector))
		self.units['inputgate'].backpropagate(configure.functions['multiply'](self.history['inputactivator'].pop(), outputvector))
		outputvector = self.units['inputactivator'].backpropagate(configure.functions['multiply'](self.history['inputgate'].pop(), outputvector))
		return outputvector

	def timingsetup(self):
		'''
			Method to prepare layer for time recurrence
		'''
		self.history['hidden'] = [numpy.zeros((self.dimensions['outputs'], 1), dtype = float)]
		self.history['deltahidden'] = [numpy.zeros((self.dimensions['outputs'], 1), dtype = float)]

class SimpleLSTM(LongShortTermMemory):
	'''
		Simple Long Short Term Memory Layer
		Mathematically, i(x(t)) = sigma2(W * x(t) + b)
						og(x(t)) = sigma1(Wo * x(t) + bo)
						fg(x(t)) = 1
						ig(x(t)) = sigma1(Wi * x(t) + bi)
	'''
	def __init__(self, inputs, outputs, alpha = None, gateactivations = None, inputtransfer = None, outputtransfer = None):
		'''
			Constructor
			: param inputs : dimension of input feature space
			: param outputs : dimension of output feature space
			: param alpha : learning rate constant hyperparameter
			: param gateactivations : sigma1, as given in its mathematical equation
			: param inputtransfer : sigma2, as given in its mathematical equation
			: param outputtransfer : sigma3, as given in its mathematical equation
		'''
		LongShortTermMemory.__init__(self, inputs, outputs, alpha)
		self.units['inputgate'] = layer.Nonlinear(self.dimensions['inputs'], self.dimensions['outputs'], alpha, gateactivations)
		self.units['forgetgate'] = connector.Constant(self.dimensions['inputs'], self.dimensions['outputs'], 1.0)
		self.units['outputgate'] = layer.Nonlinear(self.dimensions['inputs'], self.dimensions['outputs'], alpha, gateactivations)
		self.units['inputactivator'] = layer.Nonlinear(self.dimensions['inputs'], self.dimensions['outputs'], alpha, inputtransfer)
		self.units['outputactivator'] = outputtransfer(self.dimensions['outputs']) if outputtransfer is not None else transfer.Sigmoid(self.dimensions['outputs'])
		self.history['inputgate'] = list()
		self.history['outputgate'] = list()
		self.history['forgetgate'] = list()
		self.history['inputactivator'] = list()
		self.history['outputactivator'] = list()
		self.history['hidden'] = [numpy.zeros((self.dimensions['outputs'], 1), dtype = float)]
		self.history['deltahidden'] = [numpy.zeros((self.dimensions['outputs'], 1), dtype = float)]
		self.cleardeltas()

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		if inputvector.shape != (self.dimensions['inputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		return self.recur(inputvector, inputvector)

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		if outputvector.shape != (self.dimensions['outputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		return self.unrecur(outputvector)

class BasicLSTM(LongShortTermMemory):
	'''
		Basic Long Short Term Memory Layer
		Mathematically, i(x(t)) = sigma2(W * x(t) + b)
						og(x(t)) = sigma1(Wo * x(t) + bo)
						fg(x(t)) = sigma1(Wf * x(t) + bf)
						ig(x(t)) = sigma1(Wi * x(t) + bi)
	'''
	def __init__(self, inputs, outputs, alpha = None, gateactivations = None, inputtransfer = None, outputtransfer = None):
		'''
			Constructor
			: param inputs : dimension of input feature space
			: param outputs : dimension of output feature space
			: param alpha : learning rate constant hyperparameter
			: param gateactivations : sigma1, as given in its mathematical equation
			: param inputtransfer : sigma2, as given in its mathematical equation
			: param outputtransfer : sigma3, as given in its mathematical equation
		'''
		LongShortTermMemory.__init__(self, inputs, outputs, alpha)
		self.units['inputgate'] = layer.Nonlinear(self.dimensions['inputs'], self.dimensions['outputs'], alpha, gateactivations)
		self.units['forgetgate'] = layer.Nonlinear(self.dimensions['inputs'], self.dimensions['outputs'], alpha, gateactivations)
		self.units['outputgate'] = layer.Nonlinear(self.dimensions['inputs'], self.dimensions['outputs'], alpha, gateactivations)
		self.units['inputactivator'] = layer.Nonlinear(self.dimensions['inputs'], self.dimensions['outputs'], alpha, inputtransfer)
		self.units['outputactivator'] = outputtransfer(self.dimensions['outputs']) if outputtransfer is not None else transfer.Sigmoid(self.dimensions['outputs'])
		self.history['inputgate'] = list()
		self.history['outputgate'] = list()
		self.history['forgetgate'] = list()
		self.history['inputactivator'] = list()
		self.history['outputactivator'] = list()
		self.history['hidden'] = [numpy.zeros((self.dimensions['outputs'], 1), dtype = float)]
		self.history['deltahidden'] = [numpy.zeros((self.dimensions['outputs'], 1), dtype = float)]
		self.cleardeltas()

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		if inputvector.shape != (self.dimensions['inputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		return self.recur(inputvector, inputvector)

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		if outputvector.shape != (self.dimensions['outputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		return self.unrecur(outputvector)

class OutputFeedbackLSTM(LongShortTermMemory):
	'''
		Output Feedback Long Short Term Memory Layer
		Mathematically, i(x(t)) = sigma2(W * [x(t), f(x(t-1))] + b)
						og(x(t)) = sigma1(Wo * [x(t), f(x(t-1))] + bo)
						fg(x(t)) = sigma1(Wf * [x(t), f(x(t-1))] + bf)
						ig(x(t)) = sigma1(Wi * [x(t), f(x(t-1))] + bi)
	'''
	def __init__(self, inputs, outputs, alpha = None, gateactivations = None, inputtransfer = None, outputtransfer = None):
		'''
			Constructor
			: param inputs : dimension of input feature space
			: param outputs : dimension of output feature space
			: param alpha : learning rate constant hyperparameter
			: param gateactivations : sigma1, as given in its mathematical equation
			: param inputtransfer : sigma2, as given in its mathematical equation
			: param outputtransfer : sigma3, as given in its mathematical equation
		'''
		LongShortTermMemory.__init__(self, inputs, outputs, alpha)
		self.units['inputgate'] = layer.Nonlinear(self.dimensions['inputs'] + self.dimensions['outputs'], self.dimensions['outputs'], alpha, gateactivations)
		self.units['forgetgate'] = layer.Nonlinear(self.dimensions['inputs'] + self.dimensions['outputs'], self.dimensions['outputs'], alpha, gateactivations)
		self.units['outputgate'] = layer.Nonlinear(self.dimensions['inputs'] + self.dimensions['outputs'], self.dimensions['outputs'], alpha, gateactivations)
		self.units['inputactivator'] = layer.Nonlinear(self.dimensions['inputs'] + self.dimensions['outputs'], self.dimensions['outputs'], alpha, inputtransfer)
		self.units['outputactivator'] = outputtransfer(self.dimensions['outputs']) if outputtransfer is not None else transfer.Sigmoid(self.dimensions['outputs'])
		self.history['inputgate'] = list()
		self.history['outputgate'] = list()
		self.history['forgetgate'] = list()
		self.history['inputactivator'] = list()
		self.history['outputactivator'] = list()
		self.history['output'] = [numpy.zeros((self.dimensions['outputs'], 1), dtype = float)]
		self.history['hidden'] = [numpy.zeros((self.dimensions['outputs'], 1), dtype = float)]
		self.history['deltahidden'] = [numpy.zeros((self.dimensions['outputs'], 1), dtype = float)]
		self.cleardeltas()

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		if inputvector.shape != (self.dimensions['inputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		return self.recur(numpy.concatenate([inputvector, self.history['output'][-1]]), numpy.concatenate([inputvector, self.history['output'][-1]]))

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		if outputvector.shape != (self.dimensions['outputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		outputvector, hiddenvector = numpy.split(self.unrecur(outputvector), [self.dimensions['inputs']])
		return outputvector

class PeepholeLSTM(LongShortTermMemory):
	'''
		Output Feedback Long Short Term Memory Layer
		Mathematically, i(x(t)) = sigma2(W * [x(t), f(x(t-1))] + b)
						og(x(t)) = sigma1(Wo * [x(t), f(x(t-1)), h(t-1)] + bo)
						fg(x(t)) = sigma1(Wf * [x(t), f(x(t-1)), h(t-1)] + bf)
						ig(x(t)) = sigma1(Wi * [x(t), f(x(t-1)), h(t-1)] + bi)
	'''
	def __init__(self, inputs, outputs, alpha = None, gateactivations = None, inputtransfer = None, outputtransfer = None):
		'''
			Constructor
			: param inputs : dimension of input feature space
			: param outputs : dimension of output feature space
			: param alpha : learning rate constant hyperparameter
			: param gateactivations : sigma1, as given in its mathematical equation
			: param inputtransfer : sigma2, as given in its mathematical equation
			: param outputtransfer : sigma3, as given in its mathematical equation
		'''
		LongShortTermMemory.__init__(self, inputs, outputs, alpha)
		self.units['inputgate'] = layer.Nonlinear(self.dimensions['inputs'] + 2 * self.dimensions['outputs'], self.dimensions['outputs'], alpha, gateactivations)
		self.units['forgetgate'] = layer.Nonlinear(self.dimensions['inputs'] + 2 * self.dimensions['outputs'], self.dimensions['outputs'], alpha, gateactivations)
		self.units['outputgate'] = layer.Nonlinear(self.dimensions['inputs'] + 2 * self.dimensions['outputs'], self.dimensions['outputs'], alpha, gateactivations)
		self.units['inputactivator'] = layer.Nonlinear(self.dimensions['inputs'] + self.dimensions['outputs'], self.dimensions['outputs'], alpha, inputtransfer)
		self.units['outputactivator'] = outputtransfer(self.dimensions['outputs']) if outputtransfer is not None else transfer.Sigmoid(self.dimensions['outputs'])
		self.history['inputgate'] = list()
		self.history['outputgate'] = list()
		self.history['forgetgate'] = list()
		self.history['inputactivator'] = list()
		self.history['outputactivator'] = list()
		self.history['output'] = [numpy.zeros((self.dimensions['outputs'], 1), dtype = float)]
		self.history['hidden'] = [numpy.zeros((self.dimensions['outputs'], 1), dtype = float)]
		self.history['deltahidden'] = [numpy.zeros((self.dimensions['outputs'], 1), dtype = float)]
		self.cleardeltas()

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		if inputvector.shape != (self.dimensions['inputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		return self.recur(numpy.concatenate([inputvector, self.history['output'][-1]]), numpy.concatenate([inputvector, self.history['output'][-1], self.history['hidden'][-1]]))

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		if outputvector.shape != (self.dimensions['outputs'], 1):
			self.dimensionsError(self.__class__.__name__)
		outputvector, hiddenvector = numpy.split(self.unrecur(outputvector), [self.dimensions['inputs']])
		return outputvector
