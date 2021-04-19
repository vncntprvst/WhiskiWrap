
from setuptools import setup

setup(
   name='WhiskiWrap',
   version='1.0',
   author='',
   author_email='',
   packages=['WhiskiWrap','wwutils'],
   # scripts=['bin/script1','bin/script2'],
   url='http://pypi.python.org/pypi/WhiskiWrap/',
   license='LICENSE.txt',
   description='whiski package wrapper created by cxrodgers',
   long_description=open('README.md').read(),
   install_requires=['tables','pandas','MediaInfo'],
)
