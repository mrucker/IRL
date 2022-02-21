from distutils.core import setup

setup(name='irl',
      version='0.1',
      description='IRL Models',
      author='Mark Rucker',
      packages=['irl'],
      python_requires='>=3.8',
      install_requres = [
          'pandas>=1.0',
          'numpy>=1.18',
          'matplotlib>=3.2'
      ]
     )