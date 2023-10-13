from setuptools import setup, find_packages

setup(name='beach_walk_env',
      version='0.1',
      description='In this grid world, the agent walks at the beach and may be pushed '
                  'in a random direction by strong winds.',
      url='https://github.com/mschweizer/beach-walk-env',
      author='Marvin Schweizer',
      author_email='schweizer@kit.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'gym-minigrid==1.0.3',
          'gym==0.21',
          'matplotlib==3.5.1',
          'seals',
          'numpy',
      ],
      tests_require=[
          'pytest',
          'pytest-cov',
      ],
      include_package_data=True,
      python_requires='>=3.7',
      )
