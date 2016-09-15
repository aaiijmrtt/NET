'''
	Module containing Classification Dataset Loaders.
'''
import random
import numpy
from . import tools

def datasetiris(shuffle = None):
	'''
		Method to return the Iris Dataset
		: param shuffle : parameter to control whether the dataset is randomly shuffled
		: returns : Iris Dataset
		The dataset was obtained from http://archive.ics.uci.edu/ml/machine-learning-databases/iris/
	'''
	shuffle = shuffle if shuffle is not None else False
	inputs = 4
	outputs = 3
	dataset = list()
	url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
	filename = 'iris.data'
	tools.download(url, filename)
	classnames = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
	with open(filename, 'r') as datafile:
		for line in datafile:
			if line == '\n':
				continue
			features = line.strip().split(',')
			inputvector = numpy.array(features[: -1], dtype = float).reshape((inputs, 1))
			outputvector = numpy.zeros((outputs, 1), dtype = float)
			outputvector[classnames[features[-1]]][0] = 1.0
			dataset.append([inputvector, outputvector])
	if shuffle:
		random.shuffle(dataset)
	return dataset

def datasetMNIST(shuffle = None):
	'''
		Method to return the MNIST Dataset
		: param shuffle : parameter to control whether the dataset is randomly shuffled
		: returns : MNIST Dataset
		The dataset was obtained from https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/
	'''
	shuffle = shuffle if shuffle is not None else False
	inputs = 64
	outputs = 10
	dataset = list()
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra'
	filename = 'mnist.train'
	tools.download(url, filename)
	with open(filename, 'r') as datafile:
		for line in datafile:
			features = line.strip().split(',')
			inputvector = numpy.array(features[: -1], dtype = float).reshape((inputs, 1))
			outputvector = numpy.zeros((outputs, 1), dtype = float)
			outputvector[int(features[-1])][0] = 1.0
			dataset.append([inputvector, outputvector])
	url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes'
	filename = 'mnist.test'
	tools.download(url, filename)
	with open(filename, 'r') as datafile:
		for line in datafile:
			features = line.strip().split(',')
			inputvector = numpy.array(features[: -1], dtype = float).reshape((inputs, 1))
			outputvector = numpy.zeros((outputs, 1), dtype = float)
			outputvector[int(features[-1])][0] = 1.0
			dataset.append([inputvector, outputvector])
	if shuffle:
		random.shuffle(dataset)
	return dataset
