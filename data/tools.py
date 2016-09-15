'''
	Module containing tools for handling datasets.
'''
import os, urllib
import numpy, PIL.Image

def download(url, filename):
	'''
		Method to download a file
		: param url : URL of file
		: param filename : name of file on disk
	'''
	if not os.path.exists(filename):
		urllib.urlretrieve(url, filename)

def binaryvector(integer, size):
	'''
		Method to convert integer to binary vector
		: param integer : integer to be converted
		: param size : length of binary vector
		: returns : binary vector
	'''
	binary = [int(x) for x in bin(integer)[2: ]]
	binary = [0] * (size - len(binary)) + binary
	return numpy.reshape(binary, (size, 1))

def imagevector(filename):
	'''
		Method to load image as vector
		: param filename : name of file on disk
		: returns : image vector and dimensions of original image
	'''
	image = PIL.Image.open(filename)
	image.load()
	data = numpy.asarray(image, dtype = int)
	rows, columns, channels = data.shape
	vector = numpy.empty((rows * columns * channels, 1), dtype = float)
	for i in range(rows):
		for j in range(columns):
			for k in range(channels):
				vector[(k * rows + i) * columns + j][0] = data[i][j][k]
	return vector, (rows, columns, channels)

def standardization(array):
	'''
		Method to standardize a list of vectors
		: param array : list of vectors
		: returns : standardized list of vectors
	'''
	mean = numpy.mean(array, axis = 0)
	std = numpy.std(array, axis = 0)
	for i in range(len(array)):
		array[i] = numpy.divide(numpy.subtract(array[i], mean), std)
	return array

def normalization(array):
	'''
		Method to normalize a list of vectors
		: param array : list of vectors
		: returns : normalize list of vectors
	'''
	minimum = numpy.amin(array, axis = 0)
	maximum = numpy.amax(array, axis = 0)
	for i in range(len(array)):
		array[i] = numpy.divide(numpy.subtract(array[i], minimum), numpy.subtract(maximum, minimum))
	return array

def pairedstandardization(arraypairs):
	'''
		Method to standardize a list of pairs of vectors
		: param array : list of pairs of vectors
		: returns : standardized list of pairs of vectors
	'''
	primaryarray, secondaryarray = zip(*arraypairs)
	primaryarray = standardization(list(primaryarray))
	secondaryarry = standardization(list(secondaryarray))
	return zip(primaryarray, secondaryarray)

def pairednormalization(arraypairs):
	'''
		Method to normalize a list of pairs of vectors
		: param array : list of pairs of vectors
		: returns : normalized list of pairs of vectors
	'''
	primaryarray, secondaryarray = zip(*arraypairs)
	primaryarray = normalization(list(primaryarray))
	secondaryarray = normalization(list(secondaryarray))
	return zip(primaryarray, secondaryarray)
