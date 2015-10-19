'''
	Module containing Logical Function Generators.
'''
import numpy, random
from . import tools

def datasetand(size = None, variables = None, shuffle = None):
	'''
		Method to generate dataset of logical and function
		: param size : number of samples in dataset
		: param variables : dimension of input space
		: param shuffle : parameter to control whether the dataset is randomly shuffled
		: returns : dataset of logical and function
	'''
	size = size if size is not None else 100
	shuffle = shuffle if shuffle is not None else False
	variables = variables if variables is not None else 2
	count = 2 ** variables
	dataset = list()
	for i in range(size):
		x = tools.binaryvector(i % count, variables)
		y = numpy.empty((1, 1))
		y[0][0] = 1.0 if numpy.sum(x) == variables else 0.0
		dataset.append((x, y))
	if shuffle:
		random.shuffle(dataset)
	return dataset

def datasetor(size = None, variables = None, shuffle = None):
	'''
		Method to generate dataset of logical or function
		: param size : number of samples in dataset
		: param variables : dimension of input space
		: param shuffle : parameter to control whether the dataset is randomly shuffled
		: returns : dataset of logical or function
	'''
	size = size if size is not None else 100
	shuffle = shuffle if shuffle is not None else False
	variables = variables if variables is not None else 2
	count = 2 ** variables
	dataset = list()
	for i in range(size):
		x = tools.binaryvector(i % count, variables)
		y = numpy.empty((1, 1))
		y[0][0] = 1.0 if numpy.sum(x) > 0.0 else 0.0
		dataset.append((x, y))
	if shuffle:
		random.shuffle(dataset)
	return dataset

def datasetnand(size = None, variables = None, shuffle = None):
	'''
		Method to generate dataset of logical nand function
		: param size : number of samples in dataset
		: param variables : dimension of input space
		: param shuffle : parameter to control whether the dataset is randomly shuffled
		: returns : dataset of logical nand function
	'''
	size = size if size is not None else 100
	shuffle = shuffle if shuffle is not None else False
	variables = variables if variables is not None else 2
	count = 2 ** variables
	dataset = list()
	for i in range(size):
		x = tools.binaryvector(i % count, variables)
		y = numpy.empty((1, 1))
		y[0][0] = 1.0 if numpy.sum(x) < variables else 0.0
		dataset.append((x, y))
	if shuffle:
		random.shuffle(dataset)
	return dataset

def datasetnor(size = None, variables = None, shuffle = None):
	'''
		Method to generate dataset of logical nor function
		: param size : number of samples in dataset
		: param variables : dimension of input space
		: param shuffle : parameter to control whether the dataset is randomly shuffled
		: returns : dataset of logical nor function
	'''
	size = size if size is not None else 100
	shuffle = shuffle if shuffle is not None else False
	variables = variables if variables is not None else 2
	count = 2 ** variables
	dataset = list()
	for i in range(size):
		x = tools.binaryvector(i % count, variables)
		y = numpy.empty((1, 1))
		y[0][0] = 1.0 if numpy.sum(x) == 0.0 else 0.0
		dataset.append((x, y))
	if shuffle:
		random.shuffle(dataset)
	return dataset

def datasetxor(size = None, variables = None, shuffle = None):
	'''
		Method to generate dataset of logical xor function
		: param size : number of samples in dataset
		: param variables : dimension of input space
		: param shuffle : parameter to control whether the dataset is randomly shuffled
		: returns : dataset of logical xor function
	'''
	size = size if size is not None else 100
	shuffle = shuffle if shuffle is not None else False
	variables = variables if variables is not None else 2
	count = 2 ** variables
	dataset = list()
	for i in range(size):
		x = tools.binaryvector(i % count, variables)
		y = numpy.empty((1, 1))
		y[0][0] = 1.0 if numpy.sum(x) % 2.0 == 0.0 else 0.0
		dataset.append((x, y))
	if shuffle:
		random.shuffle(dataset)
	return dataset
