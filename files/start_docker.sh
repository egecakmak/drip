#!/bin/bash
docker build /home/egecakmak/Desktop/pplnenggit -f  /home/egecakmak/Desktop/pplnenggit/Dockerfile -t cuda-docker-st
docker run -it -p 6006:6006 -v /home/egecakmak/Desktop/pplnenggit/pipeline:/pipeline -v /home/egecakmak/Desktop/pplnenggit/files/images:/images -v /home/egecakmak/Desktop/pplnenggit/files:/pipeline/files --gpus all cuda-docker-st