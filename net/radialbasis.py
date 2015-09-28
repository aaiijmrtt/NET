import numpy, layer

class RadialBasis(layer.Layer):

	def __init__(self, inputs, outputs, alpha = None):
		layer.Layer.__init__(self, inputs, outputs, alpha)
		self.weights = numpy.random.normal(0.0, 1.0 / numpy.sqrt(self.inputs), (self.outputs, self.inputs))
		self.biases = numpy.random.normal(0.0, 1.0 / numpy.sqrt(self.inputs), (self.outputs, 1))
		self.function = None
		self.functionderivative = None
		self.weightderivative = None
		self.biasderivative = None
		self.cleardeltas()

	def feedforward(self, inputvector):
		self.previousinput = inputvector
		self.previousradius = numpy.empty((self.outputs, 1), dtype = float)
		self.previousoutput = numpy.empty((self.outputs, 1), dtype = float)
		for i in range(self.outputs):
			centrevector = self.weights[i].reshape((self.inputs, 1))
			self.previousradius[i][0] = numpy.sum(numpy.square(numpy.subtract(self.previousinput, centrevector))) ** 0.5
			self.previousoutput[i][0] = self.function(self.previousradius[i][0], self.biases[i][0])
		return self.previousoutput

	def backpropagate(self, outputvector):
		deltainputs = numpy.zeros((self.inputs, 1), dtype = float)
		for i in range(self.outputs):
			centrevector = self.weights[i].reshape((self.inputs, 1))
			deltainputs = numpy.add(deltainputs, numpy.multiply(outputvector[i][0], self.functionderivative(self.previousradius[i][0], self.previousinput, centrevector, self.biases[i][0], self.previousoutput[i][0])))
			self.deltaweights[i] = numpy.add(self.deltaweights[i], numpy.multiply(outputvector[i][0], numpy.transpose(self.weightsderivative(self.previousradius[i][0], self.previousinput, centrevector, self.biases[i][0], self.previousoutput[i][0]))))
			self.deltabiases[i][0] += outputvector[i][0] * self.biasesderivative(self.previousradius[i][0], self.previousinput, centrevector, self.biases[i][0], self.previousoutput[i][0])
		return deltainputs

class Gaussian(RadialBasis):

	def __init__(self, inputs, outputs, alpha = None):
		RadialBasis.__init__(self, inputs, outputs, alpha)
		self.function = lambda inputradius, coefficient: numpy.exp(-inputradius / coefficient ** 2)
		self.functionderivative = lambda inputradius, inputvector, centrevector, coefficient, output: numpy.multiply(2.0 * output / (coefficient ** 2), numpy.subtract(centrevector, inputvector))
		self.weightsderivative = lambda inputradius, inputvector, centrevector, coefficient, output: numpy.multiply(2.0 * output / (coefficient ** 2), numpy.subtract(inputvector, centrevector))
		self.biasesderivative = lambda inputradius, inputvector, centrevector, coefficient, output: 2.0 * output * numpy.sum(numpy.square(numpy.subtract(inputvector, centrevector))) / (coefficient ** 3)

class MultiQuadratic(RadialBasis):

	def __init__(self, inputs, outputs, beta = None, alpha = None):
		RadialBasis.__init__(self, inputs, outputs, alpha)
		self.beta = beta if beta is not None else 0.5
		self.function = lambda inputradius, coefficient: (inputradius ** 2 + coefficient ** 2) ** self.beta
		self.functionderivative = lambda inputradius, inputvector, centrevector, coefficient, output: numpy.multiply(self.beta * (inputradius ** 2 + coefficient ** 2) ** (self.beta - 1.0) * 2.0, numpy.subtract(inputvector, centrevector))
		self.weightsderivative = lambda inputradius, inputvector, centrevector, coefficient, output: numpy.multiply(self.beta * (inputradius ** 2 + coefficient ** 2) ** (self.beta - 1.0) * 2.0, numpy.subtract(centrevector, inputvector))
		self.biasesderivative = lambda inputradius, inputvector, centrevector, coefficient, output: self.beta * (inputradius ** 2 + coefficient ** 2) ** (self.beta - 1.0) * 2.0 * coefficient

class InverseMultiQuadratic(RadialBasis):

	def __init__(self, inputs, outputs, beta = None, alpha = None):
		RadialBasis.__init__(self, inputs, outputs, alpha)
		self.beta = beta if beta is not None else 0.5
		self.function = lambda inputradius, coefficient: (inputradius ** 2 + coefficient ** 2) ** (-self.beta)
		self.functionderivative = lambda inputradius, inputvector, centrevector, coefficient, output: numpy.multiply(-self.beta * (inputradius ** 2 + coefficient ** 2) ** (-self.beta - 1.0) * 2.0, numpy.subtract(inputvector, centrevector))
		self.weightsderivative = lambda inputradius, inputvector, centrevector, coefficient, output: numpy.multiply(-self.beta * (inputradius ** 2 + coefficient ** 2) ** (-self.beta - 1.0) * 2.0, numpy.subtract(centrevector, inputvector))
		self.biasesderivative = lambda inputradius, inputvector, centrevector, coefficient, output: -self.beta * (inputradius ** 2 + coefficient ** 2) ** (-self.beta - 1.0) * 2.0 * coefficient

class ThinPlateSpine(RadialBasis):

	def __init__(self, inputs, outputs, alpha = None):
		RadialBasis.__init__(self, inputs, outputs, alpha)
		self.function = lambda inputradius, coefficient: inputradius ** 2 * numpy.log(inputradius)
		self.functionderivative = lambda inputradius, inputvector, centrevector, coefficient, output: numpy.multiply((2.0 * numpy.log(inputradius) + 1.0), numpy.subtract(inputvector, centrevector))
		self.weightsderivative = lambda inputradius, inputvector, centrevector, coefficient, output: numpy.multiply((2.0 * numpy.log(inputradius) + 1.0), numpy.subtract(centrevector, inputvector))
		self.biasesderivative = lambda inputradius, inputvector, centrevector, coefficient, output: 0.0

class Cubic(RadialBasis):

	def __init__(self, inputs, outputs, alpha = None):
		RadialBasis.__init__(self, inputs, outputs, alpha)
		self.function = lambda inputradius, coefficient: inputradius ** 3
		self.functionderivative = lambda inputradius, inputvector, centrevector, coefficient, output: numpy.multiply(3.0 * inputradius, numpy.subtract(inputvector, centrevector))
		self.weightsderivative = lambda inputradius, inputvector, centrevector, coefficient, output: numpy.multiply(3.0 * inputradius, numpy.subtract(centrevector, inputvector))
		self.biasesderivative = lambda inputradius, inputvector, centrevector, coefficient, output: 0.0

class Linear(RadialBasis):

	def __init__(self, inputs, outputs, alpha = None):
		RadialBasis.__init__(self, inputs, outputs, alpha)
		self.function = lambda inputradius, coefficient: inputradius
		self.functionderivative = lambda inputradius, inputvector, centrevector, coefficient, output: numpy.divide(numpy.subtract(inputvector, centrevector), output)
		self.weightsderivative = lambda inputradius, inputvector, centrevector, coefficient, output: numpy.divide(numpy.subtract(centrevector, inputvector), output)
		self.biasesderivative = lambda inputradius, inputvector, centrevector, coefficient, output: 0.0
