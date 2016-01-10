'''
	Module implementing monkey patches.
	Used to configure GPU settings.
'''
import numpy
from . import gpu

functions = dict()

def numericaloptimization(GPU = False):
	global functions

	functions['add'] = gpu.add if GPU else numpy.add
	functions['amax'] = gpu.amax if GPU else numpy.amax
	functions['amin'] = gpu.amin if GPU else numpy.amin
	functions['argmin'] = gpu.argmin if GPU else numpy.argmin
	functions['argmax'] = gpu.argmax if GPU else numpy.argmax
	functions['divide'] = gpu.divide if GPU else numpy.divide
	functions['dot'] = gpu.dot if GPU else numpy.dot
	functions['exp'] = gpu.exp if GPU else numpy.exp
	functions['fabs'] = gpu.fabs if GPU else numpy.fabs
	functions['norm'] = gpu.norm if GPU else numpy.linalg.norm
	functions['log'] = gpu.log if GPU else numpy.log
	functions['mean'] = gpu.mean if GPU else numpy.mean
	functions['multiply'] = gpu.multiply if GPU else numpy.multiply
	functions['sign'] = gpu.sign if GPU else numpy.sign
	functions['sqrt'] = gpu.sqrt if GPU else numpy.sqrt
	functions['square'] = gpu.square if GPU else numpy.square
	functions['subtract'] = gpu.subtract if GPU else numpy.subtract
	functions['sum'] = gpu.sum if GPU else numpy.sum
	functions['tanh'] = gpu.tanh if GPU else numpy.tanh
	functions['transpose'] = gpu.transpose if GPU else numpy.transpose
	functions['vectorize'] = gpu.vectorize if gpu else numpy.vectorize

def reconfigure(GPU = False):
	numericaloptimization(GPU)

reconfigure(False)
