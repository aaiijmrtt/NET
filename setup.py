#!/usr/bin/python
import distutils.core

distutils.core.setup(name = 'net',
	version = '0.1',
	description = 'Python Neural Network Library',
	long_description = '''Putting Neural Networks together should be easier
		than it usually is. The code in this repository presents simple Python
		modules designed to make prototyping neural network architectures
		quick.''',
	author = 'Amitrajit Sarkar',
	author_email = 'aaiijmrtt@gmail.com',
	url = 'https://github.com/aaiijmrtt/NET',
	license = 'MIT',
	classifiers=[
		'Development Status :: 3 - Alpha',
		'Intended Audience :: Developers',
		'Topic :: Machine Learning :: Neural Networks',
		 'License :: OSI Approved :: MIT License',
		'Programming Language :: Python :: 2',
		'Programming Language :: Python :: 3'
	],
	packages = ['net', 'bench', 'data'],
	install_requires = ['numpy', 'matplotlib'],
	test_suite = ['test']
)
