import json
import argparse
import random
import os
import sys
import stat
import collections
import preprocess_images
import json_to_tfrecord
import common_utils
import datetime
import zipfile
import shutil
import subprocess


def check_prev_session():
    # Checks if a session folder already exists, if so checks its contents.
    if os.path.isdir('./dl'):
        if os.path.isfile('./dl/sessionfile'):
            os.chdir('./dl')
            with open('sessionfile') as session_file:
                session = json.load(session_file)
                date = session['session_date']
                session_files = session['files']
                print('A previous training session that was created on ' + date + ' was found.')
                for each in session_files:
                    root = each[0] + '/'
                    directories = each[1]
                    files = each[2]
                    not_existing_dirs = list(filter(lambda x: not os.path.isdir(root + x), directories))
                    not_existing_files = list(filter(lambda x: not os.path.isfile(root + x), files))
                    if len(not_existing_dirs) > 0:
                        print('Below directories under directory ' + root + ' does not exist.')
                        print(*not_existing_dirs, sep=",\n")
                    if len(not_existing_files) > 0:
                        print('Below files under directory ' + root + ' does not exist.')
                        print(*not_existing_files, sep=",\n")
            os.chdir('../')
            session_menu()


def session_menu():
    # Prints a menu of options to the user if an existing session folder is found.
    choice = input("""
                1. Resume the session found.
                2. Start a new session with the inputted parameters. (THIS WILL MOVE THE PREVIOUS SESSION FOLDER TO THE TRASH FOLDER WITH A DATE LABEL.)
                3. Exit.
                Select an option to continue : """)

    if choice == '1':
        resume_training()
        exit(0)
    elif choice == '2':
        return
    elif choice == '3':
        exit(0)
    else:
        session_menu()


def resume_training():
    print('Resuming existing training...')
    # move_contents_to_trash()
    start_training()


def save_json(data, file_name):
    print('Creating file: ' + file_name)
    x = {
        "frames": data
    }

    with open(file_name, 'w') as outfile:
        json.dump(x, outfile, indent=4)


def augment_imgs(json_path, images_path):
    preprocess_images.start_preprocessing(json_path, images_path, False)


def generate_tfrecord(json_file, images_path):
    tfrecord = json_to_tfrecord.start_conversion(json_file, images_path)
    return tfrecord


def start_pipeline(files, tfrecords):
    create_files(files, tfrecords)
    start_training()


def create_files(files, tfrecords, batch_size, number_of_steps):
    move_contents_to_trash()

    print('Creating necessary files.')

    os.mkdir('./dl/modeldir')

    network_zip_file_name = 'network.zip'
    with zipfile.ZipFile('./files/' + network_zip_file_name, 'r') as zip_file:
        zip_file.extractall('./dl')
        network_folder_name = zip_file.namelist()[0]

    for file in files:
        shutil.move(file, './dl')
        print('./' + file + ' has been moved to ' + './dl/' + file + '.')

    for tf in tfrecords.values():
        shutil.move(tf, './dl')
        print('./' + tf + ' has been moved to ' + './dl/' + tf + '.')

    shutil.copy2('./files/' + 'class_map.pbtxt', './dl')

    os.chdir('./dl')
    fine_tune_checkpoint = '/pipeline/dl/' + network_folder_name + 'model.ckpt'
    label_map_path = '/pipeline/dl/' + '/class_map.pbtxt'
    train_input = '/pipeline/dl/' + tfrecords['train']
    eval_input = '/pipeline/dl/' + tfrecords['eval']

    ppln_cfg_path = network_folder_name + '/pipeline.config'

    with open(ppln_cfg_path, 'r') as cfg:
        data = cfg.read()
    # Edits the pipeline.config file for the appropiate fields.
    data = data.replace("num_classes: 90", "num_classes: " + str(common_utils.get_num_of_classes('../files/class_map.json')))
    data = data.replace("batch_size: 24", "batch_size: " + str(batch_size))
    data = data.replace("num_steps: 200000", "num_steps: " + str(number_of_steps))
    data = data.replace("PATH_TO_BE_CONFIGURED/model.ckpt", fine_tune_checkpoint)
    data = data.replace("PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt", label_map_path)
    data = data.replace("PATH_TO_BE_CONFIGURED/mscoco_train.record", train_input)
    data = data.replace("PATH_TO_BE_CONFIGURED/mscoco_val.record", eval_input)

    os.remove(ppln_cfg_path)

    with open(ppln_cfg_path, 'w') as target:
        target.write(data)

    create_session_file()

    os.chdir('../')
    print('Done adding the necessary files.')


def move_contents_to_trash():
    # Moves the dl folder with its contents to the trash with a date tag appended into its name, if such folder exists.
    if os.path.isdir(
            './dl'):  # Checks if the dl folder which we will place the files necessary for deep learning already
        # exists.
        if os.listdir(
                './dl'):  # Checks if the above folder is not empty. If so, moves the folder to the trash and creates
            # a new one.
            now = datetime.datetime.now()
            now = now.strftime("%Y-%m-%d %H:%M")
            trash_path = './trash/dl.deleted.' + now
            os.makedirs(trash_path)
            for file in os.listdir('./dl'):
                shutil.move('./dl/' + file, trash_path)
            print('./dl' + ' has been moved to ' + trash_path + '.')
    else:
        os.mkdir('./dl')


def create_session_file():
    # Creates a file in the session folder to log the contents of the folder.
    files = []
    for file in os.walk('./'):
        files.append(file)
    y = (("session_date", str(datetime.datetime.now())), ("files", files))
    y = collections.OrderedDict(y)
    with open('sessionfile', 'w') as outfile:
        json.dump(y, outfile, indent=4)


def start_training():
    network_zip_file_name = 'network.zip'
    with zipfile.ZipFile('./files/' + network_zip_file_name, 'r') as zip_file:
        network_folder_name = zip_file.namelist()[0]

    context = '#!/bin/bash\n'
    context = context + 'cd /tensorflow/models/research\n'
    context = context + 'cd /tensorflow/models/research\n'
    context = context + 'python3 /tensorflow/models/research/object_detection/model_main.py ' \
                        '--pipeline_config_path="/pipeline/dl/'
    context = context + network_folder_name
    context = context + 'pipeline.config" --model_dir="/pipeline/dl/modeldir" --sample_1_of_n_eval_examples=1 ' \
                        '--alsologtostderr & tensorboard --logdir /pipeline/dl/modeldir '

    with open('./files/start_training.sh', 'w') as outfile:
        outfile.write(context)
    # Makes sure the launcher script files has its permissions set appropriately.
    os.chmod("./files/start_training.sh", stat.S_IRWXU)
    # Runs the scripts that start the docker container which will do the training.
    subprocess.call("./files/start_training.sh")


def main():
    parser = argparse.ArgumentParser(description='Pipeline Runner')

    parser.add_argument('--src_json_path', default='./files/metadata.json', help='Path for the original json file.')

    parser.add_argument('--image_folder_path', default='/images/', help='Path for the folder containing images.')

    parser.add_argument('--eval_percentage', default=20, help='Percentage of random images that will be used for eval.')

    parser.add_argument('--resume_training', action='store_true', help='Continue training.')

    parser.add_argument('--batch_size', default=4, help='Number of batches that will be used during training.')

    parser.add_argument('--number_of_steps', default=200000, help='Number of steps that will be taken during training.')

    args = parser.parse_args()
    src_json_path = args.src_json_path
    images_path = args.image_folder_path + '/'
    eval_percentage = int(args.eval_percentage)
    resume = args.resume_training
    batch_size = args.batch_size
    number_of_steps = args.number_of_steps

    if resume:
        resume_training()
        exit(0)

    check_prev_session()

    files_required = [src_json_path, './files/class_map.pbtxt', './files/network.zip']

    # Checks if all the required files exist, if not prints an error and exits.
    does_not_exist = list(filter(lambda x: not os.path.isfile(x), files_required))

    if len(does_not_exist) > 0:
        print('Below files do not exist. Exiting.')
        print(*does_not_exist, sep=",\n")
        exit(1)

    # Converts the class map into a json file so that we can use it easily later on.
    common_utils.convert_pbtxt_to_json()

    with open(src_json_path) as json_file:
        data = json.load(json_file, object_pairs_hook=collections.OrderedDict)

    eval_img_quantity = int(len(data['frames']) * eval_percentage / 100)
    eval_imgs = random.sample(data['frames'], k=eval_img_quantity)
    train_imgs = [x for x in data['frames'] if x not in eval_imgs]

    print('Number of eval images chosen: ' + str(eval_img_quantity))
    print('Number of train images chosen: ' + str(len(train_imgs)))
    print('******************************************************\n')

    json_file_name_without_extension = common_utils.get_file_name_without_extension_from_path(src_json_path)
    new_json_file_name_eval = json_file_name_without_extension + '_eval.json'
    new_json_file_name_train = json_file_name_without_extension + '_train.json'
    new_json_file_name_train_augmented = json_file_name_without_extension + '_train_augmented.json'

    save_json(eval_imgs, new_json_file_name_eval)
    save_json(train_imgs, new_json_file_name_train)

    augment_imgs(new_json_file_name_train, images_path)

    with open(new_json_file_name_train_augmented) as json_file:
        data = json.load(json_file, object_pairs_hook=collections.OrderedDict)
        for frame in data['frames']:
            train_imgs.append(frame)
        save_json(train_imgs, new_json_file_name_train_augmented)

    # generate_tfrecord returns the file name which we will need later.
    eval_tfrecord = generate_tfrecord(new_json_file_name_eval, images_path)
    train_tfrecord = generate_tfrecord(new_json_file_name_train_augmented, images_path)

    files = [new_json_file_name_eval, new_json_file_name_train, new_json_file_name_train_augmented]
    tfrecords = {
        'eval': eval_tfrecord,
        'train': train_tfrecord
    }

    create_files(files, tfrecords, batch_size, number_of_steps)
    start_training()


if __name__ == "__main__":
    main()
