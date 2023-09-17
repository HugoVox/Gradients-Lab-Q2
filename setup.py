from distutils.core import setup
from setuptools import setup, find_packages
import os

requirement_path = "requirements.txt"
install_requires = [] # Here we'll add: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

print(install_requires)
setup(
    name='GT-Lab-Q2',
    version='0.1dev0',
    author='Team 2', 
    author_email='minhdang2032@mail.com, khoavd2003@gmail.com, ntha21122002@gmail.com',
    url='https://github.com/HugoVox/Gradients-Lab-Q2',
    packages=install_requires,
    long_description=open('README.md').read()
)