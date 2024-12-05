# Python Analytics

These classes and elements support a broad range of analytics tasks, in Python,
for GStreamer. They will work with your distribution packages for the latest 1.24
version of GStreamer.

## Requirements

There are two installation options described below: installing on your host machine, or installing with a Docker container:


### Installing on Host (Ubuntu 24)

#### Install packages

```
sudo apt update && sudo apt -y upgrade
sudo apt install -y python3-pip  python3-venv \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-base-apps \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gir1.2-gst-plugins-bad-1.0 python3-gst-1.0 gstreamer1.0-python3-plugin-loader \
    libcairo2 libcairo2-dev git nvidia-cuda-toolkit
```

#### Install venv

`python3 -m venv --system-site-packages ~/venv`


#### Clone repo (host)

`git clone https://github.com/GrokImageCompression/grok.git`

#### Update .bashrc

```
export VIRTUAL_ENV=$HOME/venv
export PATH=$VIRTUAL_ENV/bin:$PATH
export GST_PLUGIN_PATH=$HOME/src/gstreamer/subprojects/gst-python/gst_analytics/plugins
```

and then

`source ~/.bashrc`

#### Activate venv and install basic pip packages

```
source $VIRTUAL_ENV/bin/activate
pip install --upgrade pip && \
pip install pygobject torch torchvision transformers numpy black ruff
```

#### Install requirements (host)

```
cd ~/src/gstreamer/subprojects/gst-python/gst_analytics
pip install -r requirements.txt
```


#### Install CUDA

https://developer.nvidia.com/cuda-downloads


### Installing Docker container

#### Building Docker Container

#### Important Note:

This Dockerfile maps a local `gstreamer` repository containing the `gst-python` analytics elements to the container, and expects this repository to be located in `~/src` i.e.  `~/src/gstreamer`.


#### Enable GPU Support on Host

To use the host GPU in a docker container, you will need to install the nvidia container toolkit. (if running on CPU, these steps can be skipped)


```
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### Build Ubuntu 24.04 Container
`docker build -f ./Dockerfile -t ubuntu24:latest .`

#### Run Docker Container
(if running on CPU, just remove `--gpus all` from command below)

`docker run --network kafka-network -v ~/src/gstreamer/subprojects/gst-python/:/root/gst-python -it --rm --gpus all --name ubuntu24 ubuntu24:latest /bin/bash`

In the container shell, run the following

`# cd gst_analytics && pip install -r requirements.txt`

Now you should be able to inspect the `objectdetector` element:

`gst-inspect-1.0 objectdetector`

#### Docker Cleanup

If you want to purge existing docker containers and images:

```
docker container prune -f
docker image prune -a -f
```

# Using Analytics Elements

## kafkasink

### Setting up kafka network

`docker network create kafka-network`

and list networks

`docker network ls`

### Set up kafka and zookeeper

Note: setup below is for running pipelines in another docker container. If running pipeline from host, then port changes from 9092 to 29092, and broker changes
from kafka to localhost.

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

### Create test topic
```
docker exec kafka kafka-topics --create --topic test-kafkasink-topic --bootstrap-server kafka:9092 --partitions 1 --replication-factor 1
```

### list topics

`docker exec -it kafka kafka-topics --list --bootstrap-server kafka:9092`


### delete topic

`docker exec -it kafka kafka-topics --delete --topic test-topic --bootstrap-server kafka:9092`


### consume topic

`docker exec -it kafka kafka-console-consumer --bootstrap-server kafka:9092 --topic test-kafkasink-topic --from-beginning`


## Object Detection

### Some Model Names
`fasterrcnn_resnet50_fpn`
`retinanet_resnet50_fpn`

### fasterrcnn/kafka pipeline

`GST_DEBUG=4 gst-launch-1.0 multifilesrc location=data/000015.jpg ! jpegdec ! videoconvert ! videoscale ! objectdetector model-name=fasterrcnn_resnet50_fpn device=cuda batch-size=4 ! kafkasink schema-file=data/gst_analytics_object_detector.json broker=kafka:9092 topic=test-kafkasink-topic  2>&1 | grep kafkasink`

### maskrcnn pipeline

`GST_DEBUG=4 gst-launch-1.0   filesrc location=data/people.mp4 !   decodebin ! videoconvert ! videoscale ! maskrcnn device=cuda batch-size=4 model-name=maskrcnn_resnet50_fpn ! videoconvert ! objectdetectionoverlay labels-color=0xFFFF0000 object-detection-outline-color=0xFFFF0000  ! autovideosink`


### yolo with tracking pipeline

`gst-launch-1.0   filesrc location=data/soccer_tracking.mp4 !   decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! yolo model-name=yolo11m device=cuda:0 track=True ! videoconvert  !  objectdetectionoverlay labels-color=0xFFFF0000 object-detection-outline-color=0xFFFF0000 ! autovideosink`

#### and with dectionoverlay

 `gst-launch-1.0   filesrc location=data/soccer_tracking.mp4 !   decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! yolo model-name=yolo11m device=cuda:0 track=True !  analyticsoverlay ! videoconvert !  autovideosink`



### analyticsstreammux pipeline

`GST_DEBUG=4 gst-launch-1.0 analyticsstreammux name=mux  ! videoconvert ! fakesink videotestsrc ! mux. videotestsrc pattern=ball ! mux. videotestsrc pattern=snow ! mux.`


## sample whispertranscribe pipeline

### transcription with initial prompt set

`GST_DEBUG=4 gst-launch-1.0 filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! whispertranscribe device=cuda language=ko initial_prompt = "Air Traffic Control은, radar systems를,  weather conditions에, flight paths를, communication은, unexpected weather conditions가, continuous training을, dedication과, professionalism" ! fakesink`

### translation to English pipeline

`GST_DEBUG=4 gst-launch-1.0 filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! whispertranscribe device=cuda language=ko translate=yes ! fakesink`

### coquitts pipeline

`GST_DEBUG=4 gst-launch-1.0 filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! whispertranscribe device=cuda language=ko translate=yes ! coquitts device=cuda ! audioconvert ! wavenc ! filesink location=output_audio.wav`


### whisperspeechtts pipeline

`GST_DEBUG=4 gst-launch-1.0 filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! whispertranscribe device=cuda language=ko translate=yes ! whisperspeechtts device=cuda ! audioconvert ! wavenc ! filesink location=output_audio.wav`


### mariantranslate pipeline

`GST_DEBUG=4 gst-launch-1.0 filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! whispertranscribe device=cuda language=ko translate=yes ! mariantranslate device=cuda src=en target=fr ! fakesink`

Supported src/target languages:

https://huggingface.co/models?sort=trending&search=Helsinki


### whisperlive pipeline

`GST_DEBUG=4 gst-launch-1.0 filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! whisperlive device=cuda language=ko translate=yes llm-model-name="microsoft/phi-2" ! audioconvert ! wavenc ! filesink location=output_audio.wav`

## LLM pipeline

1. generate HuggingFace token

2. `huggingface-cli login`
    and pass in token

3. LLM pipeline (in this case, we use phi-2)

`GST_DEBUG=4 gst-launch-1.0 filesrc location=data/prompt_for_llm.txt !  llm device=cuda model-name="microsoft/phi-2" ! fakesink`

## stablediffusion pipeline

`GST_DEBUG=4 gst-launch-1.0 filesrc location=data/prompt_for_stable_diffusion.txt ! stablediffusion device=cuda ! pngenc ! filesink location=output_image.png`

## caption + yolo

`GST_DEBUG=4 gst-launch-1.0   filesrc location=data/soccer_tracking.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! yolo model-name=yolo11m device=cuda:0 track=True ! caption device=cuda:0 ! textoverlay !  analyticsoverlay ! videoconvert !  autovideosink`


## caption

`GST_DEBUG=4 gst-launch-1.0   filesrc location=data/soccer_tracking.mp4 ! decodebin ! videoconvert ! caption device=cuda:0 downsampled_width=320 downsampled_height=240 prompt="What is the name of the game being played?" ! textoverlay !  fakesink`


## Building PyPI Package

1. `pip install setuptools wheel twine`
2. `python setup.py sdist bdist_wheel`
3. ls dist/