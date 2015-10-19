import numpy, random
from . import tools

def datasetiris(shuffle = None):
	'''
		Method to return the Iris Dataset
		: param shuffle : parameter to control whether the dataset is randomly shuffled
		: returns : Iris Dataset
		The dataset was obtained from http://archive.ics.uci.edu/ml/machine-learning-databases/iris/
	'''
	shuffle = shuffle if shuffle is not None else False
	url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
	filename = 'iris.data'
	classnames = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
	inputs = 4
	outputs = 3
	tools.download(url, filename)
	dataset = list()
	with open(filename, 'r') as datafile:
		for line in datafile:
			if line == '\n':
				continue
			sepallength, sepalwidth, petallength, petalwidth, classname = line.strip().split(',')
			inputvector = numpy.array([[float(sepallength)], [float(sepalwidth)], [float(petallength)], [float(petalwidth)]])
			outputvector = numpy.zeros((outputs, 1), dtype = float)
			outputvector[classnames[classname]][0] = 1.0
			dataset.append((inputvector, outputvector))
	if shuffle:
		random.shuffle(dataset)
	return dataset
