from pkg_resources import parse_requirements
from setuptools import setup, find_packages
from pathlib import Path


parent_dir = Path(__file__).parent


def get_requirements(path: Path):
    with open(path) as fobj:
        return [
            str(req)
            for req in parse_requirements(fobj.read())
        ]


setup(
    name='ebony-utils',
    setup_requires=['setuptools_scm'],
    use_scm_version=dict(
        root="../..",
        relative_to=__file__,
        write_to_template='__version__ = "{version}"',
        write_to=parent_dir / "src/ebony/utils/_version.py",
    ),
    packages=find_packages('src'),
    package_dir={"": "src"},
    install_requires=get_requirements(parent_dir / "requirements.txt"),
)