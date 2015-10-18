'''
	Module containing Benchmarking Layers.
	Classes embody Parametric Layers,
	used for benchmarking optimization algorithms.
'''
import numpy
import net

class Benchmark(net.layer.Layer):
	'''
		Benchmark Layer
	'''
	def __init__(self, parameters, multivariatefunction, alpha = None):
		'''
			Constructor
			: param parameters : dimension of parameter space
			: param multivariatefunction : error function surface to be optimized
			: param alpha : learning rate constant hyperparameter
		'''
		net.layer.Layer.__init__(self, parameters, parameters, alpha)
		self.multivariatefunction = multivariatefunction
		self.parameters['parameters'] = numpy.random.normal(0.0, 1.0 / numpy.sqrt(self.inputs), (self.inputs, 1))
		self.cleardeltas()

	def backpropagate(self, outputvector):
		'''
			Method to backpropagate derivatives through the layer
			: param outputvector : derivative vector in parameter space
		'''
		self.deltaparameters['parameters'] = numpy.add(self.deltaparameters['parameters'], outputvector)

	def train(self, trainingsetsize, batch = 1, iterations = 1):
		'''
			Method to optimize parameters
			: param trainingset : supervised data set used for training
			: param batch : training minibatch size
			: param iterations : iteration threshold for termination
		'''
		self.trainingsetup()
		for i in range(iterations):
			for j in range(trainingsetsize):
				if j % batch == 0:
					self.updateweights()
					self.multivariatefunction.update()
				self.backpropagate(self.multivariatefunction.derivative(self.parameters['parameters']))
		return numpy.subtract(self.parameters['parameters'], self.multivariatefunction.minima())
