import sys, os, numpy, unittest
import net

class DerivativesTestCase(unittest.TestCase):

	conformists = None

	def setUp(self):
		self.conformists = [net.Step, net.Sigmoid, net.HardHyperbolicTangent, net.RectifiedLinearUnit, net.ParametricRectifiedLinearUnit, net.HardShrink, net.SoftShrink, net.SoftPlus, net.ShiftScale, net.HyperbolicTangent, net.SoftSign]
		self.rebels = [net.SoftMax]

	def testconformists(self):
		epsilon = 0.0001
		delta = 0.0000001
		for conformist in self.conformists:
			for i in range(1, 100):
				conformer = conformist(i)
				inputvector = numpy.random.rand(i, 1)
				conformer.feedforward(inputvector)
				derivativevector = conformer.backpropagate(numpy.ones((i, 1), dtype = float))
				deltavector = numpy.empty((i, 1), dtype = float)
				for j in range(i):
					epsilonvector = numpy.zeros((i, 1), dtype = float)
					epsilonvector[j][0] = epsilon
					deltavector[j][0] = numpy.divide(numpy.subtract(conformer.feedforward(numpy.add(inputvector, epsilonvector)), conformer.feedforward(numpy.subtract(inputvector, epsilonvector))), 2.0 * epsilon)[j][0]
				self.assertTrue(numpy.linalg.norm(numpy.subtract(deltavector, derivativevector)) < delta, 'backpropagate derivative error in class %s' %conformist)
				conformer = None

	def testrebels(self):
		epsilon = 0.0001
		delta = 0.05
		for rebel in self.rebels:
			for i in range(500, 525):
				rebeler = rebel(i)
				inputvector = numpy.random.rand(i, 1)
				rebeler.feedforward(inputvector)
				derivativevector = rebeler.backpropagate(numpy.ones((i, 1), dtype = float))
				deltavector = numpy.empty((i, 1), dtype = float)
				for j in range(i):
					epsilonvector = numpy.zeros((i, 1), dtype = float)
					epsilonvector[j][0] = epsilon
					deltavector[j][0] = numpy.divide(numpy.subtract(rebeler.feedforward(numpy.add(inputvector, epsilonvector)), rebeler.feedforward(numpy.subtract(inputvector, epsilonvector))), 2.0 * epsilon)[j][0]
				self.assertTrue(numpy.linalg.norm(numpy.subtract(deltavector, derivativevector)) < delta, 'backpropagate derivative error in class %s' %rebel)
				rebeler = None

	def tearDown(self):
		self.conformists = None
		self.rebels = None

if __name__ == '__main__':
	numpy.random.seed(1)
	suite = unittest.TestLoader().loadTestsFromTestCase(DerivativesTestCase)
	unittest.TextTestRunner(verbosity = 9).run(suite)
