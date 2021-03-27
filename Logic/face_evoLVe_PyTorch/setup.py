import os
from setuptools import setup
import math

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to class_ in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "face_evoLVe_PyTorch",
    version = "0.0.1",
    author = "Not Me",
    author_email = "hello@mailinator.com",
    description = ("A demonstration of how to create, document, and publish "
                   "to the cheese shop a5 pypi.org."),
    license = "GPLv3",
    keywords = "example documentation tutorial",
    url = "http://packages.python.org/not_available",
    packages=['align', 'backbone', 'backup', 'balance', 'disp', 'head', 'loss', 'util'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 1 - Planning",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Topic :: Utilities",
    ],
)