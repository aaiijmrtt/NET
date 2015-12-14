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
		self.inputgate = None
		self.forgetgate = None
		self.outputgate = None
		self.inputactivator = None
		self.outputactivator = None
		self.inputgatevalue = None
		self.outputgatevalue = None
		self.forgetgatevalue = None
		self.inputactivatorvalue = None
		self.outputactivatorvalue = None
		self.previoushidden = None
		self.deltahidden = None

	def recur(self, inputvector, gatevector):
		'''
			Method to recur a vector through the layer
			: param inputvector : vector in input feature space
			: param gatevector : vector in gate input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		self.previousinput.append(inputvector)
		self.inputgatevalue.append(self.inputgate.feedforward(gatevector))
		self.forgetgatevalue.append(self.forgetgate.feedforward(gatevector))
		self.outputgatevalue.append(self.outputgate.feedforward(gatevector))
		self.inputactivatorvalue.append(self.inputactivator.feedforward(self.previousinput[-1]))
		self.previoushidden.append(configure.functions['add'](configure.functions['multiply'](self.inputgatevalue[-1], self.inputactivatorvalue[-1]), configure.functions['multiply'](self.forgetgatevalue[-1], self.previoushidden[-1])))
		self.outputactivatorvalue.append(self.outputactivator.feedforward(self.previoushidden[-1]))
		self.previousoutput.append(configure.functions['multiply'](self.outputgatevalue[-1], self.outputactivatorvalue[-1]))
		return self.previousoutput[-1]

	def unrecur(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		self.previousoutput.pop()
		self.previoushidden.pop()
		self.previousinput.pop()
		self.outputgate.backpropagate(configure.functions['multiply'](self.outputactivatorvalue.pop(), outputvector))
		outputvector = configure.functions['add'](self.outputactivator.backpropagate(configure.functions['multiply'](self.outputgatevalue.pop(), outputvector)), self.deltahidden[-1])
		self.deltahidden.append(configure.functions['multiply'](self.forgetgatevalue.pop(), outputvector))
		self.forgetgate.backpropagate(configure.functions['multiply'](self.previoushidden[-1], outputvector))
		self.inputgate.backpropagate(configure.functions['multiply'](self.inputactivatorvalue.pop(), outputvector))
		outputvector = self.inputactivator.backpropagate(configure.functions['multiply'](self.inputgatevalue.pop(), outputvector))
		return outputvector

	def timingsetup(self):
		'''
			Method to prepare layer for time recurrence
		'''
		self.previoushidden = [numpy.zeros((self.outputs, 1), dtype = float)]
		self.deltahidden = [numpy.zeros((self.outputs, 1), dtype = float)]

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
		self.inputgate = layer.Nonlinear(self.inputs, self.outputs, alpha, gateactivations)
		self.forgetgate = connector.Constant(self.inputs, self.outputs, 1.0)
		self.outputgate = layer.Nonlinear(self.inputs, self.outputs, alpha, gateactivations)
		self.inputactivator = layer.Nonlinear(self.inputs, self.outputs, alpha, inputtransfer)
		self.outputactivator = outputtransfer(self.outputs) if outputtransfer is not None else transfer.Sigmoid(self.outputs)
		self.inputgatevalue = list()
		self.outputgatevalue = list()
		self.forgetgatevalue = list()
		self.inputactivatorvalue = list()
		self.outputactivatorvalue = list()
		self.previoushidden = [numpy.zeros((self.outputs, 1), dtype = float)]
		self.deltahidden = [numpy.zeros((self.outputs, 1), dtype = float)]
		self.cleardeltas()

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		return self.recur(inputvector, inputvector)

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
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
		self.inputgate = layer.Nonlinear(self.inputs, self.outputs, alpha, gateactivations)
		self.forgetgate = layer.Nonlinear(self.inputs, self.outputs, alpha, gateactivations)
		self.outputgate = layer.Nonlinear(self.inputs, self.outputs, alpha, gateactivations)
		self.inputactivator = layer.Nonlinear(self.inputs, self.outputs, alpha, inputtransfer)
		self.outputactivator = outputtransfer(self.outputs) if outputtransfer is not None else transfer.Sigmoid(self.outputs)
		self.inputgatevalue = list()
		self.outputgatevalue = list()
		self.forgetgatevalue = list()
		self.inputactivatorvalue = list()
		self.outputactivatorvalue = list()
		self.previoushidden = [numpy.zeros((self.outputs, 1), dtype = float)]
		self.deltahidden = [numpy.zeros((self.outputs, 1), dtype = float)]
		self.cleardeltas()

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		return self.recur(inputvector, inputvector)

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
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
		self.inputgate = layer.Nonlinear(self.inputs + self.outputs, self.outputs, alpha, gateactivations)
		self.forgetgate = layer.Nonlinear(self.inputs + self.outputs, self.outputs, alpha, gateactivations)
		self.outputgate = layer.Nonlinear(self.inputs + self.outputs, self.outputs, alpha, gateactivations)
		self.inputactivator = layer.Nonlinear(self.inputs + self.outputs, self.outputs, alpha, inputtransfer)
		self.outputactivator = outputtransfer(self.outputs) if outputtransfer is not None else transfer.Sigmoid(self.outputs)
		self.inputgatevalue = list()
		self.outputgatevalue = list()
		self.forgetgatevalue = list()
		self.inputactivatorvalue = list()
		self.outputactivatorvalue = list()
		self.previousoutput = [numpy.zeros((self.outputs, 1), dtype = float)]
		self.previoushidden = [numpy.zeros((self.outputs, 1), dtype = float)]
		self.deltahidden = [numpy.zeros((self.outputs, 1), dtype = float)]
		self.cleardeltas()

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		return self.recur(numpy.concatenate([inputvector, self.previousoutput[-1]]), numpy.concatenate([inputvector, self.previousoutput[-1]]))

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		outputvector, hiddenvector = numpy.split(self.unrecur(outputvector), [self.inputs])
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
		self.inputgate = layer.Nonlinear(self.inputs + 2 * self.outputs, self.outputs, alpha, gateactivations)
		self.forgetgate = layer.Nonlinear(self.inputs + 2 * self.outputs, self.outputs, alpha, gateactivations)
		self.outputgate = layer.Nonlinear(self.inputs + 2 * self.outputs, self.outputs, alpha, gateactivations)
		self.inputactivator = layer.Nonlinear(self.inputs + self.outputs, self.outputs, alpha, inputtransfer)
		self.outputactivator = outputtransfer(self.outputs) if outputtransfer is not None else transfer.Sigmoid(self.outputs)
		self.inputgatevalue = list()
		self.outputgatevalue = list()
		self.forgetgatevalue = list()
		self.inputactivatorvalue = list()
		self.outputactivatorvalue = list()
		self.previousoutput = [numpy.zeros((self.outputs, 1), dtype = float)]
		self.previoushidden = [numpy.zeros((self.outputs, 1), dtype = float)]
		self.deltahidden = [numpy.zeros((self.outputs, 1), dtype = float)]
		self.cleardeltas()

	def feedforward(self, inputvector):
		'''
			Method to feedforward a vector through the layer
			: param inputvector : vector in input feature space
			: returns : fedforward vector mapped to output feature space
		'''
		return self.recur(numpy.concatenate([inputvector, self.previousoutput[-1]]), numpy.concatenate([inputvector, self.previousoutput[-1], self.previoushidden[-1]]))

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in output feature space
			: returns : backpropagated vector mapped to input feature space
		'''
		outputvector, hiddenvector = numpy.split(self.unrecur(outputvector), [self.inputs])
		return outputvector
