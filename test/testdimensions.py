import sys, os, numpy, unittest
import net

class DimensionsTestCase(unittest.TestCase):

	singleparameterclasses = None
	doubleparameterclasses = None
	quadrupleparameterclasses = None

	def setUp(self):
		self.singleparameterclasses = [net.Step, net.Sigmoid, net.HyperbolicTangent, net.HardHyperbolicTangent, net.RectifiedLinearUnit, net.ParametricRectifiedLinearUnit, net.HardShrink, net.SoftShrink, net.SoftMax, net.SoftPlus, net.ShiftScale, net.SoftSign, net.MeanSquared, net.CrossEntropy, net.NegativeLogLikelihood, net.CrossSigmoid, net.LogSoftMax, net.KullbackLeiblerDivergence, net.CosineDistance, net.Normalizer, net.Threshold, net.StochasticThreshold, net.HopfieldNetwork]
		self.doubleparameterclasses = [net.Linear, net.Split, net.MergeSum, net.MergeProduct, net.Perceptron, net.BidirectionalAutoassociativeMemory, net.GaussianRB, net.MultiQuadraticRB, net.InverseMultiQuadraticRB, net.ThinPlateSplineRB, net.CubicRB, net.LinearRB, net.ManhattanSO, net.EuclideanSquaredSO, net.RestrictedBoltzmannMachine, net.AutoEncoder, net.SimpleLSTM, net.BasicLSTM, net.OutputFeedbackLSTM, net.PeepholeLSTM]
		self.quadrupleparameterclasses = [net.Convolutional, net.MaxPooling, net.MinPooling, net.AveragePooling]

	def testsingleparameterclasses(self):
		for singleparameterclass in self.singleparameterclasses:
			for i in range(1, 500):
				singleparameter = singleparameterclass(i)
				self.assertEqual(singleparameter.feedforward(numpy.random.rand(singleparameter.dimensions['inputs'], 1)).shape, (singleparameter.dimensions['outputs'], 1), 'feedforward dimensions error in class %s' %singleparameterclass)
				self.assertEqual(singleparameter.backpropagate(numpy.random.rand(singleparameter.dimensions['outputs'], 1)).shape, (singleparameter.dimensions['inputs'], 1), 'backpropagate dimensions error in class %s' %singleparameterclass)
				singleparameter = None

	def testdoubleparameterclasses(self):
		for doubleparameterclass in self.doubleparameterclasses:
			for i in range(1, 35):
				for j in range(1, 35):
					doubleparameter = doubleparameterclass(i, j)
					self.assertEqual(doubleparameter.feedforward(numpy.random.rand(doubleparameter.dimensions['inputs'], 1)).shape, (doubleparameter.dimensions['outputs'], 1), 'feedforward dimensions error in class %s' %doubleparameterclass)
					self.assertEqual(doubleparameter.backpropagate(numpy.random.rand(doubleparameter.dimensions['outputs'], 1)).shape, (doubleparameter.dimensions['inputs'], 1), 'backpropagate dimensions error in class %s' %doubleparameterclass)
					doubleparameter = None

	def testquadrupleparameterclasses(self):
		for quadrupleparameterclass in self.quadrupleparameterclasses:
			for i in range(1, 10):
				for j in range(1, 10):
					for k in range(1, 3):
						for l in range(1, min(i, j)):
							quadrupleparameter = quadrupleparameterclass(i, j, k, l)
							self.assertEqual(quadrupleparameter.feedforward(numpy.random.rand(quadrupleparameter.dimensions['inputs'], 1)).shape, (quadrupleparameter.dimensions['outputs'], 1), 'feedforward dimensions error in class %s' %quadrupleparameterclass)
							self.assertEqual(quadrupleparameter.backpropagate(numpy.random.rand(quadrupleparameter.dimensions['outputs'], 1)).shape, (quadrupleparameter.dimensions['inputs'], 1), 'backpropagate dimensions error in class %s' %quadrupleparameterclass)
							quadrupleparameter = None

	def testseriesclass(self):
		for i in range(1, 25):
			for j in range(1, 25):
				for k in range(1, 25):
					doubleseries = net.Series()
					doubleseries.addlayer(net.Linear(i, j))
					doubleseries.addlayer(net.Linear(j, k))
					self.assertEqual(doubleseries.feedforward(numpy.random.rand(doubleseries.dimensions['inputs'], 1)).shape, (doubleseries.dimensions['outputs'], 1), 'feedforward dimensions error in class %s' %net.Series)
					self.assertEqual(doubleseries.backpropagate(numpy.random.rand(doubleseries.dimensions['outputs'], 1)).shape, (doubleseries.dimensions['inputs'], 1), 'backpropagate dimensions error in class %s' %net.Series)
					doubleseries = None

	def testparallelclass(self):
		for i in range(1, 10):
			for j in range(1, 10):
				for k in range(1, 10):
					for l in range(1, 10):
						doubleparallel = net.Parallel()
						doubleparallel.addlayer(net.Linear(i, j))
						doubleparallel.addlayer(net.Linear(k, l))
						self.assertEqual(doubleparallel.feedforward(numpy.random.rand(doubleparallel.dimensions['inputs'], 1)).shape, (doubleparallel.dimensions['outputs'], 1), 'feedforward dimensions error in class %s' %net.Parallel)
						self.assertEqual(doubleparallel.backpropagate(numpy.random.rand(doubleparallel.dimensions['outputs'], 1)).shape, (doubleparallel.dimensions['inputs'], 1), 'backpropagate dimensions error in class %s' %net.Parallel)
						doubleparallel = None

	def testrecurrentclass(self):
		for i in range(1, 25):
			for j in range(1, 25):
				for k in range(1, min(i, j)):
					singlerecurrent = net.Recurrent(k, net.Linear(i, j))
					singlerecurrent.timingsetup()
					self.assertEqual(singlerecurrent.feedforward(numpy.random.rand(singlerecurrent.dimensions['inputs'], 1)).shape, (singlerecurrent.dimensions['outputs'], 1), 'feedforward dimensions error in class %s' %net.Recurrent)
					self.assertEqual(singlerecurrent.backpropagate(numpy.random.rand(singlerecurrent.dimensions['outputs'], 1)).shape, (singlerecurrent.dimensions['inputs'], 1), 'backpropagate dimensions error in class %s' %net.Recurrent)
					singlerecurrent = None

	def tearDown(self):
		self.singleparameterclasses = None
		self.doubleparameterclasses = None

if __name__ == '__main__':
	suite = unittest.TestLoader().loadTestsFromTestCase(DimensionsTestCase)
	unittest.TextTestRunner(verbosity = 9).run(suite)
