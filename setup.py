import os
from setuptools import setup, find_packages

def parse_requirements(filename):
    """Load requirements from a pip requirements file"""
    with open(filename, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line and not line.startswith("#")]

def parse_extra_requirements(directory):
    """Load extra requirements from multiple files"""
    extras = {}
    if os.path.exists(directory):
        for req_file in os.listdir(directory):
            if req_file.endswith(".txt"):  # Only process .txt files
                name = os.path.splitext(req_file)[0]  # e.g., 'language' from 'language.txt'
                extras[name] = parse_requirements(os.path.join(directory, req_file))
    return extras

setup(
    name="gst-python-ml",
    version="0.2.0",
    packages=find_packages(where="plugins/python"),
    package_dir={"": "plugins/python"},
    include_package_data=True,
    description="An ML package for GStreamer",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="LGPL-3.0",
    license_files=["COPYING"],
    author="Aaron Boxer",
    author_email="aaron.boxer@collabora.com",
    url="https://github.com/collabora/gst-python-ml",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=parse_requirements("requirements.txt"),  # Base requirements
    extras_require=parse_extra_requirements("requirements"),  # Grouped dependencies
)
