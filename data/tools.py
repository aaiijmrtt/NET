'''
	Module containing tools for handling datasets.
'''
import os, urllib
import numpy, Image

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
		Method to convert image to vector
		: param filename : name of file on disk
		: returns : image vector and dimensions of original image
	'''
	image = Image.open(filename)
	image.load()
	data = np.asarray(img, dtype = int)
	rows, columns, channels = data.shape
	vector = numpy.empty((rows * columns * channels, 1), dtype = float)
	for i in range(rows):
		for j in range(columns):
			for k in range(channels):
				vector[(k * rows + i) * columns + j][0] = data[i][j][k]
	return vector, (rows, columns, channels)
