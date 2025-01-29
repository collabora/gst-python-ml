## Profiling DeepStream on Docker

0. if you have not already done so, generate an api token after
logging in to `https://org.ngc.nvidia.com/setup`

1. login with nvidia token

`$ docker login nvcr.io`

```
Username: $oauthtoken
Password:  $YOUR_API_TOKEN
```

2. pull docker container

`$ docker pull nvcr.io/nvidia/deepstream:7.1-triton-multiarch`

3. run docker container

```
$ docker run --gpus all -it --rm --network=host \
  nvcr.io/nvidia/deepstream:7.1-triton-multiarch
```

4. list TensorRT engines

`$ find /opt/nvidia/deepstream -name "*.engine"`

5. edit config file

`$ vi /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/source1_file_dec_infer_resnet_int8.txt`

a) set engine file:

`model-engine-file=/opt/nvidia/deepstream/deepstream-7.1/samples/models/Primary_Detector/resnet18_trafficcamnet_pruned.onnx_b1_gpu0_int8.engine`

b) set loop to true

c) add this section for profiling

```
[application]
enable-perf-measurement=1
perf-measurement-interval-sec=5
```

6. run pipeline

`$ deepstream-app -c /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/source1_file_dec_infer_resnet_int8.txt`


6. optional : commit changes

```
docker ps
docker commit $CONTAINER_ID deepstream-7.1-custom
docker run --gpus all -it --rm --network=host deepstream-7.1-custom
```
