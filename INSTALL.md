# GStreamer Python ML Installation

There are two installation options described below: installing on your host machine,
or installing with a Docker container:

### Host Install (Ubuntu 24)

#### Install packages

```
sudo apt update && sudo apt -y upgrade
sudo apt install -y python3-pip  python3-venv \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-base-apps \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gir1.2-gst-plugins-bad-1.0 python3-gst-1.0 gstreamer1.0-python3-plugin-loader \
    libcairo2 libcairo2-dev git
```

#### Install venv

`python3 -m venv --system-site-packages ~/venv`


#### Clone repo (host)

`git clone https://github.com/collabora/gst-python-ml.git`

#### Update .bashrc

```
export VIRTUAL_ENV=$HOME/venv
export PATH=$VIRTUAL_ENV/bin:$PATH
export GST_PLUGIN_PATH=$HOME/src/gst-python-ml/plugins
```

and then

`source ~/.bashrc`

#### Activate venv and install basic pip packages

```
source $VIRTUAL_ENV/bin/activate
pip install --upgrade pip
```

#### Install pip requirements

```
cd ~/src/gst-python-ml
pip install -r requirements.txt
```

### Docker Install

#### Build Docker Container

Important Note:

This Dockerfile maps a local `gst-python-ml` repository to the container,
and expects this repository to be located in `~/src` i.e.  `~/src/gst-python-ml`.


#### Enable Docker GPU Support on Host

To use the host GPU in a docker container, you will need to install the nvidia container toolkit. If running on CPU, these steps can be skipped.


Add nvidia repository (Ubuntu)

```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

Then

```
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### Build Ubuntu 24.04 Container
`docker build -f ./Dockerfile -t ubuntu24:latest .`

#### Run Docker Container

a) If running on CPU, just remove `--gpus all` from command below
b) This command assumes you have set up a Kafka network as described below

`docker run -v ~/src/gst-python-ml/:/root/gst-python-ml -it --rm --gpus all --name ubuntu24 ubuntu24:latest /bin/bash`

In the container shell, run

`pip install -r requirements.txt`

to install base requirements, and then

`cd gst-python-ml` to run the pipelines below. After installing requirements,
it is recommended to open another terminal on host and run

`docker ps` to get the container id, and then run

`docker commit $CONTAINER_ID` to commit the changes, where `$CONTAINER_ID`
is the id for your docker instance.

#### Docker Cleanup

If you want to purge existing docker containers and images:

```
docker container prune -f
docker image prune -a -f
```

## IMPORTANT NOTE

To use the language elements included in this project, the `nvidia-cuda-toolkit`
ubuntu package must be installed, and additional pip requirements must be installed from
`requirements/language_requrements.txt`

## Post Install

Run `gst-inspect-1.0 python` to see all of the pyml elements listed.

# Building PyPI Package

1. `pip install setuptools wheel twine`
2. `python setup.py sdist bdist_wheel`
3. ls dist/
