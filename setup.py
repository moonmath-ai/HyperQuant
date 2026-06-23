from setuptools import setup, find_packages

setup(
    name="hyperquant",
    version="0.1.0",
    description="E8 Lattice Quantization for LLMs and Diffusion Transformers",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://moonmath.ai/hyperquant/",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
    ],
    extras_require={
        "dev": ["pytest"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
)
