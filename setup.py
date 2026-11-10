from setuptools import setup, find_packages
import os

_CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

try:
    README = open(os.path.join(_CURRENT_DIR, "README.md"), encoding="utf-8").read()
except IOError:
    README = ""

setup(
    name="pipeline",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "absl-py",
        "flax",
        "jax>=0.5",
        "jaxlib>=0.5",
        "matplotlib",
        "ml-collections",
        "numpy",
        "optax",
        "pyyaml",
        "scipy",
        "soap-jax==0.1.0+sled",   # the vendored copy: pip install ./third_party/soap_jax
        "tabulate",
        "wandb",
    ],
    extras_require={
        "testing": ["pytest"],
    },
    license="Apache-2.0",
    description=("SLED: two-phase PINN training in JAX/Flax with the frozen "
                 "second-order batch study - Sobol, "
                 "leverage, kernel-diagonal and SLED quasi-Newton batch samplers, "
                 "phase-1 hand-over checkpoints, off-batch loss tracking and a "
                 "safeguarded SSBroyden."),
    long_description=README,
    long_description_content_type="text/markdown",
)
