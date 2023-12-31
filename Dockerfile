FROM python:3.8.6

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    git

# focalnet GitHub 저장소 복제
RUN git clone https://github.com/sdk17586/focalnet.git /root/focalnet

RUN mkdir -p /root/focalnet/weight

RUN pip install --find-links https://download.pytorch.org/whl/torch_stable.html \
    Pillow \
    loguru \
    pydantic \
    GPUtil==1.4.0 \
    torch==1.12.1+cu113 \
    torchvision==0.13.1+cu113 \
    grad-cam==1.4.3 \
    torchmetrics==0.7.0 \
    pytorch_lightning==1.8.6 \
    timm==0.4.12 \
    yacs==0.1.8 \
    termcolor==1.1.0 \
    opencv-python==4.8.0.76

COPY . /app
