from distutils.core import setup
from setuptools import setup, find_packages

setup(
    name='GT-Lab-Q2',
    version='0.1dev0',
    author='Team 2', 
    author_email='author_email@mail.com',
    packages=find_packages(),
    long_description=open('README.md').read()
)