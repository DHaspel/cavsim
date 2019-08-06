#! /opt/conda/bin/python3
""" General PyPI compliant setup.py configuration of the package """

# Copyright 2019 FAU-iPAT (http://ipat.uni-erlangen.de/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from setuptools import setup
from cavsim import __author__, __version__, __license__, __email__


def get_readme() -> str:
    """
    Method to read the README.rst file

    :return: string containing README.md file
    """
    with open('README.md') as file:
        return file.read()


def get_requirements() -> list:
    """
    Method to read the env/requirements.txt and return as list

    :return: list of requirements
    """
    with open('env/requirements.txt') as file:
        return file.read().splitlines()


# ------------------------------------------------------------------------------
#   Call setup method to define this package
# ------------------------------------------------------------------------------
setup(
    name='cavsim',
    version=__version__,
    author=__author__,
    author_email=__email__,
    description='???',  # todo: add description
    long_description=get_readme(),
    url='???',  # todo: set package url
    license=__license__,
    keywords='???',  # todo: add keywords
    packages=['cavsim'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires='>=3.6',
    install_requires=get_requirements(),
    zip_safe=False)
