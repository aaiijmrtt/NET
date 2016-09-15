'''
	Module containing serialization/deserialization functions for Net objects.
'''
import json, glob, os, tarfile
import numpy
import net

def write(writable, arrays = None):
	'''
		Method to recursively write (deconstruct) nested Net object
		: param writable : nested Net object to be written
		: param arrays : list of numpy arrays in Net object
		: returns : dict representation of nested Net object, and numpy arrays in Net object
	'''
	if isinstance(writable, (type(None), int, long, float, bool)):
		returnable = writable
	elif isinstance(writable, list):
		returnable = list()
		for value in writable:
			returnedwritable, arrays = write(value, arrays)
			returnable.append(returnedwritable)
	elif isinstance(writable, dict):
		returnable = dict()
		for key, value in writable.iteritems():
			returnable[key], arrays = write(value, arrays)
	else:
		returnable = dict()
		returnable['__class__'] = writable.__class__.__name__
		if isinstance(writable, numpy.ndarray):
			if arrays is None:
				arrays = list()
			returnable['__array__'] = len(arrays)
			arrays.append(writable)
		elif isinstance(writable, net.base.Net):
			for key, value in vars(writable).iteritems():
				if key == 'functions':
					continue
				elif key == 'backpointer':
					returnable[key] = None
				else:
					returnable[key], arrays = write(value, arrays)
	return returnable, arrays

def read(readable, backpointed = None, arrays = None):
	'''
		Method to recursively read (reconstruct) nested Net object
		: param readable : dict representation of nested Net object
		: param backpointed : pointer to nesting Net object
		: param arrays : list of numpy arrays in Net object
		: returns : nested Net object to be read
	'''
	if isinstance(readable, dict) and '__class__' in readable:
		if readable['__class__'] == 'ndarray':
			returnable = arrays[readable['__array__']]
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
						returnable.__dict__[key] = read(value, returnable, arrays)
					else:
						returnable.__dict__[key] = read(value, backpointed, arrays)
			if hasattr(returnable, '__finit__'):
				getattr(returnable, '__finit__')()
	elif isinstance(readable, dict):
		returnable = dict()
		for key, value in readable.iteritems():
			if isinstance(returnable, net.base.Net):
				returnable[key] = read(value, returnable, arrays)
			else:
				returnable[key] = read(value, backpointed, arrays)
	elif isinstance(readable, list):
		returnable = list()
		for value in readable:
			if isinstance(returnable, net.base.Net):
				returnable.append(read(value, returnable, arrays))
			else:
				returnable.append(read(value, backpointed, arrays))
	elif isinstance(readable, (type(None), int, long, float, bool)):
		returnable = readable
	return returnable

def serialize(network):
	'''
		Method to serialize nested Net object
		: param network : nested Net object to be serialized
		: returns : serialized string representation of nested Net object, and list of numpy arrays in Net object
	'''
	writable, arrays = write(network)
	return json.dumps(writable, indent = 4), arrays

def deserialize(string, arrays):
	'''
		Method to deserialize nested Net object
		: param network : string representation of nested Net object
		: param arrays : list of numpy arrays in Net object
		: returns : nested Net object to be deserialized
	'''
	return read(json.loads(string), None, arrays)

def store(network, path, name):
	'''
		Method to store nested Net object to directory
		: param network : nested Net object to be stored
		: param path : name of directory on disk
		: param name : name of config file
	'''
	os.makedirs(path)
	writable, arrays = serialize(network)
	json.dump(writable, open(os.path.join(path, name + '.config'), 'w'), indent = 4)
	for i in range(len(arrays)):
		numpy.save(os.path.join(path, name + str(i)), arrays[i])

def load(path, name):
	'''
		Method to load nested Net object from file
		: param path : name of directory on disk
		: param name : name of config file
		: returns : nested Net object to be loaded
	'''
	arrays = [numpy.load(os.path.join(path, name + str(i) + '.npy')) for i in range(len(glob.glob(os.path.join(path, name + '*.npy'))))]
	return deserialize(json.load(open(os.path.join(path, name + '.config'), 'r')), arrays)

def compress(network, path, name):
	'''
		Method to store nested Net object to directory
		: param network : nested Net object to be stored
		: param path : name of directory on disk
		: param name : name of tarfile on disk
	'''
	store(network, path, name)
	with tarfile.TarFile(os.path.join(path, name + '.tar'), 'w') as tar:
		tar.add(name)

def decompress(path, name):
	'''
		Method to load nested Net object from file
		: param path : name of directory on disk
		: param name : name of tarfile on disk
		: returns : nested Net object to be loaded
	'''
	with tarfile.TarFile(os.path.join(path, name + '.tar'), 'r') as tar:
		tar.extractall()
	return load(path, name)
