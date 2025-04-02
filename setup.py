import os

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as f:
  requirements = f.read().splitlines()
setup(
  name='neural-MFG',
  version='1.0.0',
  packages=find_packages(),
  install_requires=requirements,
)

# log
# [1.0.0] - 2025-04-03
# - support both conditional and unconditional generation
# - add support for dimension reduction
