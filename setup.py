from setuptools import find_packages, setup
from typing import List

hyphen_e = "-e ."


def get_requirements(file_path) -> List:

    with open(file_path) as file:
        requirements = file.readlines()

        requirements = [req.replace("\n", "") for req in requirements]

        if hyphen_e in requirements:
            requirements.remove(hyphen_e)

    return requirements


setup(
    name='Lookalike',
    version='0.0.1',
    author='JR Jahed',
    author_email='jrjahed100@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
