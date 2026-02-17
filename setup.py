"""
Setup script for MTech Thesis package.

Allows installation of the src/ package for easier imports.
"""

from setuptools import setup, find_packages

setup(
    name="xai-container-malware",
    version="0.1.0",
    author="MTech Student",
    author_email="your.email@iiitg.ac.in",
    description="Explainable AI for Docker Container Malware Detection",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/prelim-thesis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.9",
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if not line.startswith("#") and line.strip()
    ],
)
