from pathlib import Path
from pkg_resources import parse_requirements
from setuptools import find_namespace_packages, setup

parent_dir = Path(__file__).parent

def get_requirements(path: Path):
    with open(path) as fobj:
        return [str(req) for req in parse_requirements(fobj.read())]

setup(
    name="aika-ml",
    setup_requres=["setuptools_scm"],
    use_scm_version=dict(
        root="../..",
        relative_to=__file__,
        write_to_template='"__version__"= "{version}"',
        write_to=parent_dir / "src/aika/ml/_version.py",
    ),
    packages=find_namespace_packages("src", include=["aika.*"]),
    package_dir={"": "src"},
    install_requres=[
        "pandas>=1.2.0",
        "numpy",
        "aika-time",
        "aika-utilities",
        "sklearn>=1.1.2"
    ],
    extras_requires=dict(
        test=["pytest", "black", "isort"]
    )
)