#!/bin/bash
apt-get update 
apt-get install -y git nano unzip wget 
apt-get clean 
apt-get autoremove
apt-get install -y python3 python python-scipy python-pip python3-pip python-setuptools python3-setuptools python3-scipy 
apt-get -y upgrade 
apt-get install -y protobuf-compiler python-pil python-lxml python-tk python3-pil python3-lxml python3-tk cuda9.0 cuda-cublas-9-0 cuda-cufft-9-0 cuda-curand-9-0 cuda-cusolver-9-0 cuda-cusparse-9-0 libnccl2=2.2.13-1+cuda9.0 cuda-command-line-tools-9-0 libglib2.0-0 libsm6 libxext6 
pip install pip==9.0.1 
pip3 install pip==9.0.1 
pip install tensorflow-gpu==1.12.0 Cython contextlib2 jupyter matplotlib 
pip3 install tensorflow-gpu==1.12.0 Cython contextlib2 jupyter matplotlib absl-py opencv-python albumentations 
cd /usr/local/cuda 
mkdir include 
cd /install 
tar -xzvf cudnn-9.0-linux-x64-v7.5.1.10.tgz 
cp cuda/include/cudnn.h /usr/local/cuda/include/cudnn.h 
cp cuda/lib64/libcudnn* /usr/local/cuda/lib64 
chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn* 
cd / 
mkdir tensorflow 
git clone https://github.com/cocodataset/cocoapi.git 
cd /cocoapi/PythonAPI 
python3 setup.py build_ext --inplace 
rm -rf build 
cd /tensorflow 
git clone https://github.com/tensorflow/models.git 
cp -r /cocoapi/PythonAPI/pycocotools /tensorflow/models/research 
cd /tensorflow/models/research 
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip 
unzip protobuf.zip 
./bin/protoc object_detection/protos/*.proto --python_out=. 
python3 setup.py build 
python3 setup.py install 
echo "export PYTHONPATH=$PYTHONPATH:/tensorflow/models/research:/tensorflow/models/research/slim" >> /root/.bashrc 
cd / 
rm -rf install rm -rf cocoapi 
apt-get remove -y python-numpy python3-numpy 
pip install numpy==1.16.1 
pip3 install numpy==1.16.1 scipy
