'''
	Module containing serialization/deserialization functions for Net objects.
'''
import json
import numpy
import net

def write(writable):
	'''
		Method to recursively write (deconstruct) nested Net object
		: param writable : nested Net object to be written
		: returns : dict representation of nested Net object
	'''
	if isinstance(writable, (type(None), int, long, float, bool)):
		returnable = writable
	elif isinstance(writable, list):
		returnable = list()
		for value in writable:
			returnable.append(write(value))
	elif isinstance(writable, dict):
		returnable = dict()
		for key, value in writable.iteritems():
			returnable[key] = write(value)
	else:
		returnable = dict()
		returnable['__class__'] = writable.__class__.__name__
		if isinstance(writable, numpy.ndarray):
			returnable['__array__'] = numpy.ndarray.tolist(writable)
		elif isinstance(writable, net.base.Net):
			for key, value in vars(writable).iteritems():
				if key == 'functions':
					continue
				elif key == 'backpointer':
					returnable[key] = None
				else:
					returnable[key] = write(value)
	return returnable

def read(readable, backpointed = None):
	'''
		Method to recursively read (reconstruct) nested Net object
		: param readable : dict representation of nested Net object
		: param backpointed : pointer to nesting Net object
		: returns : nested Net object to be read
	'''
	if isinstance(readable, dict) and '__class__' in readable:
		if readable['__class__'] == 'ndarray':
			returnable = numpy.array(readable['__array__'])
		else:
			classbyname = getattr(net, readable['__class__'])
			returnable = classbyname.__new__(classbyname)
			for key, value in readable.iteritems():
				if key in ['__class__', 'functions']:
					continue
				elif key == 'backpointer':
					returnable.__dict__[key] = backpointed
				else:
					if isinstance(returnable, net.base.Net):
						returnable.__dict__[key] = read(value, returnable)
					else:
						returnable.__dict__[key] = read(value, backpointed)
			if hasattr(returnable, '__finit__'):
				getattr(returnable, '__finit__')()
	elif isinstance(readable, dict):
		returnable = dict()
		for key, value in readable.iteritems():
			if isinstance(returnable, net.base.Net):
				returnable[key] = read(value, returnable)
			else:
				returnable[key] = read(value, backpointed)
	elif isinstance(readable, list):
		returnable = list()
		for value in readable:
			if isinstance(returnable, net.base.Net):
				returnable.append(read(value, returnable))
			else:
				returnable.append(read(value, backpointed))
	elif isinstance(readable, (type(None), int, long, float, bool)):
		returnable = readable
	return returnable

def serialize(network):
	'''
		Method to serialize nested Net object
		: param network : nested Net object to be serialized
		: returns : serialized string representation of nested Net object
	'''
	return json.dumps(network, default = write, indent = 4)

def deserialize(string):
	'''
		Method to deserialize nested Net object
		: param network : string representation of nested Net object
		: returns : nested Net object to be deserialized
	'''
	return read(json.loads(string))

def store(network, filename):
	'''
		Method to store nested Net object to file
		: param network : nested Net object to be stored
	'''
	with open(filename, 'w') as fileout:
		json.dump(network, fileout, default = write, indent = 4)

def load(filename):
	'''
		Method to load nested Net object from file
		: param filename : name of file on disk
		: returns : nested Net object to be loaded
	'''
	with open(filename, 'r') as filein:
		network = read(json.load(filein))
	return network
