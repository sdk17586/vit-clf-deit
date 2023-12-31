# vit-clf-deit-classification

## docker_image_Build
```bat
docker build -t dnn_test .
```
## Container
```bat
docker run -it \
    --name dnn_test \
    --privileged \
    --gpus all \
    --network host \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /etc/localtime:/etc/localtime:ro \
    -e TZ=Asia/Seoul \
    -v /dev:/dev \
    -w /root \
    dnn_test:latest
```
## container_setting
```bat
docker exec -it dnn_test bash

git clone https://github.com/sdk17586/vit-clf-deit.git

mkdir -p /root/vit-clf-deit/weight

```

## training
```bat
python3 /root/vit-clf-deit/bin/train.py
```

## predict
```bat
python3 /root/vit-clf-deit/bin/predict.py
```

