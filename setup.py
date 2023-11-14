from setuptools import setup, find_packages

setup(name='beach_walk_env',
      version='0.2',
      description='In this grid world, the agent walks at the beach and may be pushed '
                  'in a random direction by strong winds.',
      url='https://github.com/mschweizer/beach-walk-env',
      author='Marvin Schweizer',
      author_email='schweizer@kit.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'minigrid',
          'gymnasium',
          'matplotlib==3.5.1',
          'seals',
          'numpy',
          'stable-baselines3',
      ],
      tests_require=[
          'pytest',
          'pytest-cov',
      ],
      include_package_data=True,
      python_requires='>=3.10',
      )
