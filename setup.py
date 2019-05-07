from setuptools import setup, find_packages

setup(
   name='SpecAugment',
   version='1.2.5',
   description='A implementation of "SpecAugment"',
   url              = 'https://github.com/shelling203/SpecAugment',
   packages         = find_packages(exclude = ['docs', 'tests*']),
   install_requires=['librosa', 'matplotlib'], #external packages as dependencies
)