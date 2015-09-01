import sys, os
sys.path.append(os.path.abspath('..'))

import net, numpy, unittest

class DerivativesTestCase(unittest.TestCase):

	singleparameterclasses = None

	def setUp(self):
		self.singleparameterclasses = [net.Step, net.Sigmoid, net.HardHyperbolicTangent, net.RectifiedLinearUnit, net.ParametricRectifiedLinearUnit, net.HardShrink, net.SoftShrink, net.SoftPlus, net.ShiftScale]

	def testsingleparameterclasses(self):
		epsilon = 0.0001
		for singleparameterclass in self.singleparameterclasses:
			for i in range(1, 100):
				singleparameter = singleparameterclass(i)
				inputvector = numpy.random.rand(i, 1)
				singleparameter.feedforward(inputvector)
				derivativevector = singleparameter.backpropagate(numpy.ones((i, 1), dtype = float))
				deltavector = numpy.empty((i, 1), dtype = float)
				for j in range(i):
					epsilonvector = numpy.zeros((i, 1), dtype = float)
					epsilonvector[j][0] = epsilon
					deltavector[j][0] = numpy.divide(numpy.subtract(singleparameter.feedforward(numpy.add(inputvector, epsilonvector)), singleparameter.feedforward(numpy.subtract(inputvector, epsilonvector))), 2.0 * epsilon)[j][0]
				self.assertTrue(numpy.linalg.norm(numpy.subtract(deltavector, derivativevector)) < 0.00000001, 'backpropagate derivative error in class %s' %singleparameterclass)
				singleparameter = None

	def tearDown(self):
		self.singleparameterclasses = None

if __name__ == '__main__':
	numpy.random.seed(1)
	suite = unittest.TestLoader().loadTestsFromTestCase(DerivativesTestCase)
	unittest.TextTestRunner(verbosity = 9).run(suite)
