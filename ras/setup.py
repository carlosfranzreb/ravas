from setuptools import setup


NAME = "stream_processing"
DESCRIPTION = "Framework for processing audio and video streams."
URL = "https://github.com/carlosfranzreb/stream_processing"


def req_file(filename: str) -> list[str]:
    """Get requirements from file"""
    with open(filename, encoding="utf-8") as f:
        content = f.readlines()
    required, links = list(), list()
    for line in content:
        line = line.strip()
        required.append(line)
    return required, links


REQUIRED = req_file("requirements.txt")
EXTRAS = {}
VERSION = "0.3.0"

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    url=URL,
    packages=["stream_processing"],
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
)
