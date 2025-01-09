FROM ubuntu:latest

# Set environment variables to non-interactive
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

# Update and install dependencies
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y python3-pip  python3-venv \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-base-apps \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gir1.2-gst-plugins-bad-1.0 python3-gst-1.0 libcairo2 libcairo2-dev \
    git nvidia-cuda-toolkit gstreamer1.0-python3-plugin-loader

# Create and activate a virtual environment with access to system packages
RUN python3 -m venv --system-site-packages /opt/venv
RUN echo 'source /opt/venv/bin/activate' >> /root/.bashrc
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Activate the virtual environment and install necessary Python packages
RUN /bin/bash -c "\
    source /opt/venv/bin/activate && \
    pip install --upgrade pip && \
    pip install pygobject torch torchvision transformers numpy black ruff"

# Set some environment variables
ENV GST_PLUGIN_PATH=/root/gst-python-ml/plugins
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib

# allow Python to properly handle Unicode characters during logging.
ENV PYTHONIOENCODING=utf-8

COPY requirements.txt /root/

# Set the working directory (optional)
WORKDIR /root

# Set the entry point to bash for interactive use
CMD ["/bin/bash"]
