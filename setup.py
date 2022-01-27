from distutils.core import setup

setup(name='combat',
      version='0.5',
      description='Models of Combat Games',
      author='Mark Rucker',
      packages=['combat'],
      python_requires='>=3.8',
      install_requres = [
          'pandas>=1.0',
          'numpy>=1.18',
          'matplotlib>=3.2'
      ]
     )