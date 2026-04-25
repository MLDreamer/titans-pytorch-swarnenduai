from setuptools import setup, find_packages

setup(
    name="titans_pytorch_swarnenduai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
    ],
    author="Swarnendu",
    author_email="swarnenduiitb2020@gmail.com",
    description="Robust implementation of Google's Titans: Learning to Memorize at Test Time",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MLDreamer/titans-pytorch-swarnenduai",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
