"""
AdamPID - Professional PID Controller and Autotuner
Based on QuickPID and sTune libraries by dlloydev
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="adampid",
    version="1.0.0",
    author="AdamPID Contributors",
    description="Professional PID controller and autotuner with inflection point method",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/adampid",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[],
    extras_require={
        "plotting": ["matplotlib>=3.5.0"],
        "enhanced": ["numpy>=1.20.0", "matplotlib>=3.5.0"],
        "dev": ["pytest>=6.0", "black", "isort", "flake8"],
    },
)
