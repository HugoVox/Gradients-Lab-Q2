from distutils.core import setup
from setuptools import setup

setup(
    name='GT-Lab-Q2',
    version='0.1dev0',
    author='Team 2', 
    author_email='minhdang2032@mail.com, khoavd2003@gmail.com, ntha21122002@gmail.com',
    packages=['sentence_transformers',
              'transformers',
              'psycopg2',
              'gradio',
              'torch',
              'peft'],
    long_description=open('README.md').read()
)