import sys, os
sys.path.append(os.path.abspath('..'))

import net, numpy, unittest

class SampleCodeTestCase(unittest.TestCase):

	def testsamplecode(self):

		n_input = 2
		n_hidden = 6
		n_output = 1

		myseriesnet = net.Series()

		myseriesnet.addlayer(net.Linear(n_input, n_hidden))
		myseriesnet.addlayer(net.HyperbolicTangent(n_hidden))
		myseriesnet.addlayer(net.Linear(n_hidden, n_output))
		myseriesnet.addlayer(net.MeanSquared(n_output))

		myseriesnet.applyvelocity(0.9)
		myseriesnet.applylearningrate(0.025)

		mytrainingset = list()

		for i in range(500):

			x = numpy.zeros((n_input, 1), dtype = float)
			y = numpy.zeros((n_output, 1), dtype = float)

			if i % 4 == 0:
				mytrainingset.append((x, y))
			elif i % 4 == 1:
				x[1][0] = 1.0
				y[0][0] = 1.0
				mytrainingset.append((x, y))
			elif i % 4 == 2:
				x[0][0] = 1.0
				y[0][0] = 1.0
				mytrainingset.append((x, y))
			else:
				x[0][0] = 1.0
				x[1][0] = 1.0
				mytrainingset.append((x, y))

		mytestingset = mytrainingset

		myoptimizer = net.Optimizer(myseriesnet, mytrainingset, mytestingset)
		myoptimizer.train()

		self.assertTrue(myoptimizer.test()[0][0] < 0.001, 'unreliable sample code accuracy')

if __name__ == '__main__':
	suite = unittest.TestLoader().loadTestsFromTestCase(SampleCodeTestCase)
	unittest.TextTestRunner(verbosity = 9).run(suite)
