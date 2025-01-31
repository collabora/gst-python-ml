from setuptools import setup, find_packages

def parse_requirements(filename):
    """Load requirements from a pip requirements file"""
    with open(filename, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line and not line.startswith("#")]

setup(
    name="gst-python-ml",
    version="0.1.0",
    packages=find_packages(where="plugins/python"),
    package_dir={"": "plugins/python"},
    include_package_data=True,
    description="An ML package for GStreamer",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Aaron Boxer",
    author_email="aaron.boxer@collabora.com",
    url="https://github.com/collabora/gst-python-ml",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: LGPL License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=parse_requirements("requirements.txt"),
)
