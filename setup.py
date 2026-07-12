from setuptools import setup, find_packages

setup(
    name="futures-foundation-model",
    version="2.0.0",
    description="Futures-market foundation layer on pretrained Chronos-Bolt: frozen embeddings + strategy-pluggable training/eval pipelines",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    url="https://github.com/johnamcruz/Futures-Foundation-Model",
    packages=find_packages(),
    python_requires=">=3.9",
    # Core install is torch-free (the parent process must never load torch —
    # see futures_foundation/foundation.py). Torch/Chronos run only inside
    # the embed subprocess; install them via the [foundation] extra.
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "scikit-learn>=1.3",
    ],
    extras_require={
        "foundation": ["torch>=2.0", "chronos-forecasting"],
        "heads": ["xgboost>=2.0", "joblib>=1.3"],
        "regime": ["hmmlearn>=0.3"],   # futures_foundation.regime market-state HMM
        # futures_foundation.rl default PPO trainer — lazy-imported at train
        # time only; importing the rl package needs none of these.
        "rl": ["stable-baselines3>=2.0", "gymnasium>=0.29"],
        "onnx": ["onnxmltools", "skl2onnx"],
        "dev": ["pytest>=7.0", "black", "ruff", "hmmlearn>=0.3"],
    },
)
