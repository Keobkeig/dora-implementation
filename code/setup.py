from setuptools import find_packages, setup

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="dora-implementation",
    version="0.1.0",
    author="DoRA Implementation Team",
    author_email="dora@example.com",
    description=(
        "A PyTorch implementation of DoRA: " "Weight-Decomposed Low-Rank Adaptation from scratch"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/dora-implementation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.0.0",
        ],
        "visualization": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dora-train=scripts.train_commonsense:main",
            "dora-benchmark=scripts.run_benchmarks:main",
        ],
    },
    include_package_data=True,
    package_data={
        "configs": ["*.yaml", "*.yml", "*.json"],
    },
    keywords=[
        "deep learning",
        "machine learning",
        "transformer",
        "fine-tuning",
        "parameter efficient",
        "DoRA",
        "LoRA",
        "LLaMA",
        "vision transformer",
        "pytorch",
    ],
    project_urls={
        "Bug Reports": "https://github.com/example/dora-implementation/issues",
        "Source": "https://github.com/example/dora-implementation",
        "Documentation": "https://github.com/example/dora-implementation/docs",
        "Paper": "https://arxiv.org/abs/2402.09353",
    },
)
