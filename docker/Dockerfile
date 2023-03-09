FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04
# use an older system (18.04) to avoid opencv incompatibility (issue#3524)

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo ninja-build
RUN ln -sv /usr/bin/python3 /usr/bin/python

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/pip/3.6/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install PyDrive2
RUN pip install --user tensorboard cmake   # cmake from apt-get is too old
RUN pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html 
RUN pip install cython pyyaml==5.1
RUN pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# install detectron2:
RUN pip install detectron2==0.1.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html
# RUN pip install opencv-python
RUN pip install labelbox
RUN pip install simplejson
RUN pip install progressbar
#import torch, torchvision
#print(torch.__version__, torch.cuda.is_available())
#!gcc --version
RUN pip install google-cloud-storage


#new
RUN pip install scikit-image
RUN pip install matplotlib
RUN pip install pycocotools

#new for testing objects
RUN pip install geojson
RUN pip install shapely
RUN pip install packaging
RUN pip install pyproj
RUN pip install pygeotile

#new for mask to polygon
# website I used - https://rocreguant.com/convert-a-mask-into-a-polygon-for-images-using-shapely-and-rasterio/1786/
RUN pip install rasterio

#new for analyze model
RUN pip install open3d
RUN pip install potpourri3d
RUN pip install vedo

WORKDIR /opt/project
#WORKDIR /opt/project/mal-coral
