from setuptools import setup, find_packages

setup(
    name='tello_vision_control',
    version='1.0.0',
    description='Python package made for controlling a tello drone with computer vision',
    author='Guilhem Schena',
    packages=find_packages(),
    install_requires=[
    "numpy"],
)