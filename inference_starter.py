import argparse
import subprocess
import os
import stat
import shutil

parser = argparse.ArgumentParser(description='Inference Pipeline Starter')

parser = argparse.ArgumentParser(description='Recognize objects in a given set of images using the model trained.')

# parser.add_argument('--graph', default='./ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb',
#                     help='Path for the trained '
#                          'graph.')

parser.add_argument('--images_path', default='./files/images', help='Path for the folder containing image folders.')

# parser.add_argument('--label_map_path', default='./class_map.pbtxt', help='Path for the label map file.')

# parser.add_argument('--results_path', default='./images', help='Path where the resulting json files will be saved.')

parser.add_argument('--threshold', default=60, help='Threshold.')

args = parser.parse_args()
# graph = args.graph
images_path = args.images_path + '/'
# label_map_path = args.label_map_path + '/'
# results_path = args.results_path + '/'
threshold = args.threshold

path = os.path.dirname(os.path.abspath(__file__))
images_abs_path = os.path.abspath(images_path)

# arg_1 = '--graph ' + str(graph) + ' '
# arg_2 = '--images_path ' + images_path + ' '
# arg_3 = '--label_map_path ' + label_map_path + ' '
# arg_4 = '--results_path ' + results_path + ' '
arg_5 = '--threshold ' + str(threshold)

pipeline_starter_command = "python3 object_recognition.py " + arg_5

startup_context = ""
startup_context = startup_context + "#!/bin/bash\n"
startup_context = startup_context + "export PYTHONPATH=:/tensorflow/models/research:/tensorflow/models/research/slim\n"
startup_context = startup_context + "cd /pipeline/files\n"
startup_context = startup_context + "cp -f json_to_tfrecord.py preprocess_images.py pipeline_runner.py common_utils.py object_recognition.py \
 ..\n"
startup_context = startup_context + "cd /pipeline\n"
startup_context = startup_context + pipeline_starter_command

with open('./files/startup.sh', 'w') as outfile:
    outfile.write(startup_context)

set_docker_context = ""
set_docker_context = set_docker_context + "#!/bin/bash\n"
set_docker_context = set_docker_context + "docker build " + path + " -f  " + path + "/Dockerfile -t cuda-docker-st\n"
set_docker_context = set_docker_context + "docker run -it -p 6006:6006 -v " + path + "/pipeline:/pipeline " "-v " + \
                     images_abs_path + ":/images " + "-v " + path + "/files:/pipeline/files "
set_docker_context = set_docker_context + "--gpus all cuda-docker-st"

# set_docker_context = ""
# set_docker_context = set_docker_context + "#!/bin/bash\n"
# set_docker_context = set_docker_context + "docker build " + path + " -f  " + path + "/Dockerfile -t cuda-docker-st\n"
# set_docker_context = set_docker_context + "docker run -it -p 6006:6006 -v " + path + "/dl:/pipeline/dl " "-v " + \
#                      images_abs_path + ":/images " + "-v " + path + "/files:/pipeline/files " + \
#                      "-v " + path + "/trash:/pipeline/trash " + session_arg
# set_docker_context = set_docker_context + "--gpus all cuda-docker-st"

with open('./files/start_docker.sh', 'w') as outfile:
    outfile.write(set_docker_context)

os.chmod('./files/startup.sh', stat.S_IRWXU)
os.chmod('./files/start_docker.sh', stat.S_IRWXU)

subprocess.call(["./files/start_docker.sh", images_path])
