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

### Object Detection

Possible model names:
`fasterrcnn_resnet50_fpn`
`retinanet_resnet50_fpn`

#### fasterrcnn/kafka

`GST_DEBUG=4 gst-launch-1.0 multifilesrc location=data/000015.jpg ! jpegdec ! videoconvert ! videoscale ! pyml_objectdetector model-name=fasterrcnn_resnet50_fpn device=cuda batch-size=4 ! pyml_kafkasink schema-file=data/pyml_object_detector.json broker=kafka:9092 topic=test-kafkasink-topic  2>&1 | grep pyml_kafkasink`

#### maskrcnn

`GST_DEBUG=4 gst-launch-1.0   filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! pyml_maskrcnn device=cuda batch-size=4 model-name=maskrcnn_resnet50_fpn ! videoconvert ! objectdetectionoverlay labels-color=0xFFFF0000 object-detection-outline-color=0xFFFF0000  ! autovideosink`


#### yolo with tracking

`gst-launch-1.0   filesrc location=data/soccer_tracking.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! pyml_yolo model-name=yolo11m device=cuda:0 track=True ! videoconvert  !  pyml_overlay labels-color=0xFFFF0000 object-detection-outline-color=0xFFFF0000 ! autovideosink`

#### yolo with overlay

 `gst-launch-1.0   filesrc location=data/soccer_tracking.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! pyml_yolo model-name=yolo11m device=cuda:0 track=True !  pyml_overlay ! videoconvert !  autovideosink`


### streammux pipeline

`GST_DEBUG=4 gst-launch-1.0 pyml_streammux name=mux  ! videoconvert ! fakesink videotestsrc ! mux. videotestsrc pattern=ball ! mux. videotestsrc pattern=snow ! mux.`


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


