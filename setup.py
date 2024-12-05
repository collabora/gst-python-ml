from setuptools import setup, find_packages
import os

def parse_requirements(filename):
    """Load requirements from a pip requirements file"""
    with open(filename, "r") as file:
        return [line.strip() for line in file if line and not line.startswith("#")]

# Helper function to include all .py files manually
def find_py_files(directory):
    py_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.relpath(os.path.join(root, file), directory))
    return py_files

setup(
    name="gstreamer-analytics",  # Package name
    version="1.24.0",  # Version
    packages=['gst_analytics', 'gst_analytics.engine'],  # Manually specify packages
    package_dir={"gst_analytics": "plugins/python"},  # Map everything to gst_analytics
    package_data={  # Manually include all .py files
        "gst_analytics": find_py_files("plugins/python"),
    },
    include_package_data=True,  # Ensure all data files are included
    description="An analytics package for GStreamer",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Aaron Boxer",
    author_email="aaron.boxer@collabora.com",
    url="https://gitlab.freedesktop.org/gstreamer/gstreamer",  # Project URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: LGPL License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=parse_requirements("requirements.txt"),
)
