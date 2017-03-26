from setuptools import setup, find_packages
from persist import __version__ as VERSION


def get_requirements():
    req = open('requirements.txt').readlines()
    req = map(lambda c: c.rstrip('\n'), req)
    return req


REQUIREMENTS = get_requirements()

setup(name='persist',
      version=VERSION,
      description='Implement persistent collections for dask',
      url='https://github.com/eserie/persist',
      author='eserie',
      author_email='eserie@gmail.com',
      license='',
      packages=find_packages(),
      install_requires=REQUIREMENTS,
      )
