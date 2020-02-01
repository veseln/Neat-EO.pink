"""
Neat-EO.pink
----------
High Performance Computer Vision framework for GeoSpatial Imagery

<http://www.Neat-EO.pink>
"""

from setuptools import setup, find_packages
from os import path
import re

here = path.dirname(__file__)

with open(path.join(here, "neat_eo", "__init__.py"), encoding="utf-8") as f:
    version = re.sub("( )*__version__( )*=( )*", "", f.read()).replace('"', "")

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(path.join(here, "requirements.txt")) as f:
    install_requires = f.read().splitlines()

setup(
    name="neat-eo.pink",
    version=version,
    url="https://github.com/datapink/Neat-EO.pink",
    download_url="https://github.com/datapink/Neat-EO.pink/releases",
    license="MIT",
    maintainer="DataPink",
    maintainer_email="hello@datapink.com",
    description="High Performances Computer Vision framework for GeoSpatial Imagery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests"]),
    install_requires=install_requires,
    entry_points={"console_scripts": ["neo = neat_eo.tools.__main__:main"]},
    include_package_data=True,
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
    ],
)
