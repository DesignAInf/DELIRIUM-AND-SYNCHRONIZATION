from setuptools import setup, find_packages

setup(
    name="icu-delirium-pomdp",
    version="1.0.0",
    author="Luca M. Possati",
    description="Two-agent POMDP model of ICU delirium based on active inference",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/possati/icu-delirium-pomdp",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "icu-delirium=main:main",
        ],
    },
)
