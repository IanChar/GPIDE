from distutils.core import setup
from setuptools import find_packages

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='gpide',
    version='0.0.1',
    packages=find_packages(),
    license='MIT License',
    install_requires=requirements,
)
