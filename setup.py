from setuptools import setup, find_packages

setup(
    name="feathermap",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "matplotlib",
        "streamlit"
        # "pathlib",
        # "sklearn",
        # "ipython",
        # "ipdb",
    ],
)
