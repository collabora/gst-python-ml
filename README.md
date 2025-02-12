# GStreamer Python ML

This project provides Python base classes and GStreamer elements supporting a broad range
of ML features. 

Supported functionality includes:

1. object detection
1. tracking
1. video captioning
1. translation
1. transcription
1. speech to text
1. text to speech
1. text to image
1. LLMs
1. serializing model metadata to Kafka server

ML toolkits are supported via the `MLEngine` abstraction - we have nominal support for
TensorFlow, LiteRT and OpenVINO, but all testing thus far has been done with PyTorch.

These elements will work with your distribution's GStreamer packages. They have been tested on Ubuntu 24 with GStreamer 1.24.

## Python Version

All elements have been tested with Python 3.12, the installed version of Python on Ubuntu 24

## Install

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


## Using GStreamer Python ML Elements

## Pipelines

Below are some sample pipelines for the various elements in this project.

### kafkasink

#### Setting up kafka network

`docker network create kafka-network`

and list networks

`docker network ls`

#### docker launch

To launch a docker instance with the kafka network, add ` --network kafka-network  `
to the docker launch command above.

#### Set up kafka and zookeeper

Note: setup below assumes you are running your pipeline in a docker container. 
If running pipeline from host, then the port changes from `9092` to `29092`,
and the broker changes from `kafka` to `localhost`.

```
docker stop kafka zookeeper
docker rm kafka zookeeper
docker run -d --name zookeeper --network kafka-network -e ZOOKEEPER_CLIENT_PORT=2181 confluentinc/cp-zookeeper:latest
docker run -d --name kafka --network kafka-network \
  -e KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181 \
  -e KAFKA_ADVERTISED_LISTENERS=INSIDE://kafka:9092,OUTSIDE://localhost:29092 \
  -e KAFKA_LISTENER_SECURITY_PROTOCOL_MAP=INSIDE:PLAINTEXT,OUTSIDE:PLAINTEXT \
  -e KAFKA_LISTENERS=INSIDE://0.0.0.0:9092,OUTSIDE://0.0.0.0:29092 \
  -e KAFKA_INTER_BROKER_LISTENER_NAME=INSIDE \
  -e KAFKA_BROKER_ID=1 \
  -e KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1 \
  -p 9092:9092 \
  -p 29092:29092 \
  confluentinc/cp-kafka:latest
```

#### Create test topic
```
docker exec kafka kafka-topics --create --topic test-kafkasink-topic --bootstrap-server kafka:9092 --partitions 1 --replication-factor 1
```

#### list topics

`docker exec -it kafka kafka-topics --list --bootstrap-server kafka:9092`


#### delete topic

`docker exec -it kafka kafka-topics --delete --topic test-topic --bootstrap-server kafka:9092`


#### consume topic

`docker exec -it kafka kafka-console-consumer --bootstrap-server kafka:9092 --topic test-kafkasink-topic --from-beginning`


### non ML

`GST_DEBUG=4 gst-launch-1.0 videotestsrc ! video/x-raw,width=1280,height=720 ! pyml_overlay meta-path=data/sample_metadata.json tracking=true ! videoconvert ! autovideosink`

Note: make sure to set the following in `.bashrc` file :

`export GST_PLUGIN_PATH=/home/$USER/src/gst-python-ml/plugins:$GST_PLUGIN_PATH`


### Bird's Eye View

`GST_DEBUG=4 gst-launch-1.0 filesrc location=data/soccer_single_camera.mp4 ! decodebin ! videoconvert ! pyml_birdseye ! videoconvert ! autovideosink`

`GST_DEBUG=4 gst-launch-1.0 filesrc location=data/soccer_single_camera.mp4 ! decodebin ! videorate ! video/x-raw,framerate=30/1 ! videoconvert ! pyml_birdseye ! videoconvert ! openh264enc ! h264parse ! matroskamux ! filesink location=output.mkv`

### Classification

```
GST_DEBUG=4 gst-launch-1.0  filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! pyml_classifier model-name=resnet18 device=cuda !  videoconvert !  autovideosink
```


### Object Detection

#### TorchVision

`pyml_objectdetector` supports all TorchVision  object detection models.
Simply choose a suitable model name and set it on the `model-name` property.
A few possible model names:

```
fasterrcnn_resnet50_fpn
ssdlite320_mobilenet_v3_large
```

##### fasterrcnn

`GST_DEBUG=4 gst-launch-1.0  filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! pyml_objectdetector model-name=fasterrcnn_resnet50_fpn device=cuda batch-size=4 ! videoconvert ! pyml_overlay ! videoconvert ! autovideosink`

##### fasterrcnn/kafka

`GST_DEBUG=4 gst-launch-1.0  filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! pyml_objectdetector model-name=fasterrcnn_resnet50_fpn device=cuda batch-size=4 ! pyml_kafkasink schema-file=data/pyml_object_detector.json broker=kafka:9092 topic=test-kafkasink-topic  2>&1 | grep pyml_kafkasink`

#### maskrcnn

`GST_DEBUG=4 gst-launch-1.0   filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! pyml_maskrcnn device=cuda batch-size=4 model-name=maskrcnn_resnet50_fpn ! videoconvert ! pyml_overlay ! autovideosink`


#### yolo with tracking

`gst-launch-1.0   filesrc location=data/soccer_tracking.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! videoconvert ! pyml_yolo model-name=yolo11m device=cuda:0 track=True ! pyml_overlay  ! videoconvert ! autovideosink`

#### yolo with overlay

 `gst-launch-1.0   filesrc location=data/soccer_tracking.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! pyml_yolo model-name=yolo11m device=cuda:0 track=True !  pyml_overlay ! videoconvert !  autovideosink`


### streammux/streamdemux pipeline

```
gst-launch-1.0     videotestsrc pattern=ball ! video/x-raw, width=320, height=240 ! queue ! pyml_streammux name=mux     videotestsrc pattern=smpte ! video/x-raw, width=320, height=240 ! queue ! mux.sink_0     videotestsrc pattern=zone-plate ! video/x-raw, width=320, height=240 ! queue ! mux.sink_1     mux.src ! queue ! pyml_streamdemux name=demux     demux.src_0 ! queue ! glimagesink     demux.src_1 ! queue ! glimagesink
```

### Transcription

#### transcription with initial prompt set

`GST_DEBUG=4 gst-launch-1.0 filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! pyml_whispertranscribe device=cuda language=ko initial_prompt = "Air Traffic Control은, radar systems를,  weather conditions에, flight paths를, communication은, unexpected weather conditions가, continuous training을, dedication과, professionalism" ! fakesink`

#### translation to English

`GST_DEBUG=4 gst-launch-1.0 filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! pyml_whispertranscribe device=cuda language=ko translate=yes ! fakesink`

#### coquitts

`GST_DEBUG=4 gst-launch-1.0 filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! pyml_whispertranscribe device=cuda language=ko translate=yes ! pyml_coquitts device=cuda ! audioconvert ! wavenc ! filesink location=output_audio.wav`


#### whisperspeechtts

`GST_DEBUG=4 gst-launch-1.0 filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! pyml_whispertranscribe device=cuda language=ko translate=yes ! pyml_whisperspeechtts device=cuda ! audioconvert ! wavenc ! filesink location=output_audio.wav`


#### mariantranslate

`GST_DEBUG=4 gst-launch-1.0 filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! pyml_whispertranscribe device=cuda language=ko translate=yes ! pyml_mariantranslate device=cuda src=en target=fr ! fakesink`

Supported src/target languages:

https://huggingface.co/models?sort=trending&search=Helsinki


#### whisperlive

`GST_DEBUG=4 gst-launch-1.0 filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! pyml_whisperlive device=cuda language=ko translate=yes llm-model-name="microsoft/phi-2" ! audioconvert ! wavenc ! filesink location=output_audio.wav`

### LLM

1. generate HuggingFace token

2. `huggingface-cli login`
    and pass in token

3. LLM pipeline (in this case, we use phi-2)

`GST_DEBUG=4 gst-launch-1.0 filesrc location=data/prompt_for_llm.txt !  pyml_llm device=cuda model-name="microsoft/phi-2" ! fakesink`

### stablediffusion

`GST_DEBUG=4 gst-launch-1.0 filesrc location=data/prompt_for_stable_diffusion.txt ! pyml_stablediffusion device=cuda ! pngenc ! filesink location=output_image.png`

#### Caption

#### caption + yolo

`GST_DEBUG=4 gst-launch-1.0   filesrc location=data/soccer_tracking.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! pyml_yolo model-name=yolo11m device=cuda:0 track=True ! pyml_caption device=cuda:0 ! textoverlay !  pyml_overlay ! videoconvert !  autovideosink`


#### caption

`GST_DEBUG=4 gst-launch-1.0   filesrc location=data/soccer_tracking.mp4 ! decodebin ! videoconvert ! pyml_caption device=cuda:0 downsampled_width=320 downsampled_height=240 prompt="What is the name of the game being played?" ! textoverlay !  autovideosink`
