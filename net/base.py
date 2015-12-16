'''
	Module containing Base Class of Net.
'''
import numpy

class Net(object):
	'''
		Base Class for all Net Classes
	'''
	def dimensionsError(self, classname):
		raise RuntimeError('Unexpected Vector Dimensions in ' + classname)
