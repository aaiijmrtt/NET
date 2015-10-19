import numpy, random
from . import tools

def datasetbostonhousing(shuffle = None):
	'''
		Method to return the Boston Housing Dataset
		: param shuffle : parameter to control whether the dataset is randomly shuffled
		: returns : Boston Housing Dataset
		The dataset was obtained from http://archive.ics.uci.edu/ml/machine-learning-databases/housing/
	'''
	shuffle = shuffle if shuffle is not None else False
	url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
	filename = 'bostonhousing.data'
	inputs = 13
	outputs = 1
	tools.download(url, filename)
	dataset = list()
	with open(filename, 'r') as datafile:
		for line in datafile:
			crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat, medv = line.strip().split()
			inputvector = numpy.array([[float(crim)], [float(zn)], [float(indus)], [float(chas)], [float(nox)], [float(rm)], [float(age)], [float(dis)], [float(rad)], [float(tax)], [float(ptratio)], [float(b)], [float(lstat)]])
			outputvector = numpy.array([[float(medv)]])
			dataset.append((inputvector, outputvector))
	if shuffle:
		random.shuffle(dataset)
	return dataset
