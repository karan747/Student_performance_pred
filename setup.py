from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
# return all the list of requirments

    requirements = []
    with open(file_path) as file_obj:
        requirments = file_obj.readlines()
        requirments = [r.replace('\n','')for r in requirments]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
name = "Student_performance_prediction",
version='0.0.1',
author='Karan_bais',
author_email='karanbais2701@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirments.txt')
)