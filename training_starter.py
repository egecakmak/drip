import argparse
import subprocess
import os
import stat
import shutil

parser = argparse.ArgumentParser(description='Training Pipeline Starter')

parser.add_argument('--src_json_path', default='./files/veriler.json', help='Path for the original '
                                                                            'json file.')

parser.add_argument('--image_folder_path', default='./files/images/', help='Path for the folder containing images.')

parser.add_argument('--eval_percentage', default=20, help='Percentage of random images that will be used for eval.')

parser.add_argument('--resume_training', action='store_true', help='Continue training.')

parser.add_argument('--session_folder', default='./usersession/', help='Path for the folder containing files \
from other training session.')

parser.add_argument('--batch_size', default=4, help='Number of batches that will be used during training.')

parser.add_argument('--number_of_steps', default=200000, help='Number of steps that will be taken during training.')

args = parser.parse_args()
src_json_path = args.src_json_path
images_path = args.image_folder_path + '/'
eval_percentage = args.eval_percentage
resume = args.resume_training
session_folder = args.session_folder
batch_size = args.batch_size
number_of_steps = args.number_of_steps

path = os.path.dirname(os.path.abspath(__file__))
images_abs_path = os.path.abspath(images_path)
session_abs_path = os.path.abspath(session_folder)

if src_json_path != './files/veriler.json':
    target_path = './files/' + os.path.basename(src_json_path)
    if os.path.isfile(target_path):
        os.remove(target_path)
    shutil.copy2(src_json_path, './files')

# arg_1 = '--src_json_path ' + str(src_json_path) + ' '
# arg_2 = '--image_folder_path ' + str(images_path) + ' '
arg_3 = '--eval_percentage ' + str(eval_percentage) + ' '
arg_4 = '--resume_training ' if resume else ''
arg_5 = '--batch_size ' + str(batch_size) + ' '
arg_6 = '--number_of_steps ' + str(number_of_steps)

pipeline_starter_command = "python3 pipeline_runner.py " + arg_3 + arg_4 + arg_5 + arg_6

startup_context = ""
startup_context = startup_context + "#!/bin/bash\n"
startup_context = startup_context + "export PYTHONPATH=:/tensorflow/models/research:/tensorflow/models/research/slim\n"
startup_context = startup_context + "cd /pipeline/files\n"
startup_context = startup_context + "cp -f json_to_tfrecord.py preprocess_images.py pipeline_runner.py common_utils.py \
 ..\n"
startup_context = startup_context + "cd /pipeline\n"
startup_context = startup_context + pipeline_starter_command

with open('./files/startup.sh', 'w') as outfile:
    outfile.write(startup_context)

session_arg = '-v ' + session_abs_path + ':/pipeline/dl ' if resume else ''

set_docker_context = ""
set_docker_context = set_docker_context + "#!/bin/bash\n"
set_docker_context = set_docker_context + "docker build " + path + " -f  " + path + "/Dockerfile -t cuda-docker-st\n"
set_docker_context = set_docker_context + "docker run -it -p 6006:6006 -v " + path + "/pipeline:/pipeline " "-v " + \
                     images_abs_path + ":/images " + "-v " + path + "/files:/pipeline/files " + session_arg
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
