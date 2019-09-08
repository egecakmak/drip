FROM nvidia/cuda:9.0-base-ubuntu16.04 

COPY ./files/cudnn-9.0-linux-x64-v7.5.1.10.tgz /install/cudnn-9.0-linux-x64-v7.5.1.10.tgz
COPY ./files/setup.sh /setup.sh
RUN chmod 700 /setup.sh && ./setup.sh
COPY ./files/startup.sh /startup.sh

WORKDIR "/"
CMD bash -C '/startup.sh';'bash'


