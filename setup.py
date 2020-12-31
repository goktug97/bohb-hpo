#!/usr/bin/env python

import os
from setuptools import setup

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='BOHB_HPO',
      version=f'0.0.8',
      description='Bayesian Optimization Hyperband Hyperparameter Optimization',
      author='Göktuğ Karakaşlı',
      author_email='karakasligk@gmail.com',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/goktug97/bohb_hpo',
      packages = ['bohb'],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License"
      ],
      python_requires='>=3.7',
      install_requires=[
          'numpy',
          'scipy',
          'statsmodels'],
      include_package_data=True)
