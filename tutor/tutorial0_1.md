# Setting Up

Before you can install net, you must make sure you have the following on your
computer:

* [Python](https://www.python.org/): net is written in Python, for Python.
There are no wrappers for other languages at the moment. So you'll need that.

* [Numpy](http://www.numpy.org/): net uses Numpy structures and functions
internally. You'll need that as well.

* [MatplotLib](http://matplotlib.org/): bench requires MatplotLib for plotting
functions. You might be able to get away without using it, if you know what you
are doing, but we would still recommend that you install: it is really quite
neat in itself.

## Downloading

If you use Git, you may clone a local copy of the package:

		$ git clone https://github.com/aaiijmrtt/NET.git

Otherwise, you may simply click on the 'Download ZIP' option, and unzip locally.

## Installing

From the root directory of the package, run:

		$ python setup.py install

You may consider using the option `--record files.txt`, if you wish to
uninstall later. You may need superuser privileges.

## Uninstalling

If you installed with:

		$ python setup.py install --record files.txt

You may uninstall by running from the root directory of the package:

		$ cat files.txt | xargs rm -rf
