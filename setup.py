from setuptools import setup, find_packages

setup(
    name="smolvla-mujoco",
    version="0.1.0",
    description="Systematic Evaluation of SmolVLA in MuJoCo Simulation",
    author="Your Name",
    author_email="your@email.com",
    url="https://github.com/yourusername/smolvla-mujoco",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "lerobot",
        "mujoco>=3.0.0",
        "gymnasium>=0.29.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "isort", "flake8"],
        "vis": ["matplotlib", "seaborn", "plotly"],
        "quantize": ["bitsandbytes", "optimum", "onnxruntime"],
    },
)
