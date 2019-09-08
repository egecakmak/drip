import json
import argparse
import urllib.request
import tensorflow as tf
from common_utils import get_id_from_name, get_name_from_id, get_file_name_without_extension_from_path
import shutil

from os.path import basename, splitext, isfile, isdir
from os import makedirs, rename
from datetime import datetime
from object_detection.utils import dataset_util

flags = tf.app.flags
FLAGS = flags.FLAGS


def start_conversion(file_path, images_path):
    # Creates a trash folder to move old dl folders into it.
    if not isdir('./trash'):
        makedirs('./trash')

    json_file_name_without_extension = get_file_name_without_extension_from_path(file_path)
    tfrecord_file_name = json_file_name_without_extension + '.tfrecord'

    # Moves old tfrecord files to the trash with a date tag appended.
    if isfile(tfrecord_file_name):
        print(tfrecord_file_name + ' already exists.')
        now = datetime.now()
        now = now.strftime("%Y-%m-%d %H:%M")
        trash_path = './trash/' + tfrecord_file_name + '.deleted.' + now
        rename(tfrecord_file_name, trash_path)
        print(tfrecord_file_name + ' has been moved to ' + trash_path + '.')

    print('Creating ' + tfrecord_file_name + ' .')
    if not isfile(tfrecord_file_name):
        f = open('./' + tfrecord_file_name, "w")
        f.close()

    writer = tf.python_io.TFRecordWriter(tfrecord_file_name)
    with open(file_path) as json_file:
        data = json.load(json_file)
        for frame in data['frames']:
            print(frame)
            image_path = './images_augmented/' if '_augmented' in basename(frame['path']) else images_path
            with tf.gfile.GFile(image_path + frame['path'], 'rb') as fid:
                encoded_img = fid.read()
            tf_example = create_tf(frame, encoded_img)
            writer.write(tf_example.SerializeToString())

        writer.close()

    print('Done creating ' + tfrecord_file_name + ' .')
    return tfrecord_file_name


def get_file_type(url):
    root, ext = splitext(url)
    return ext[1:]


def create_tf(data, encoded_img):
    url = data['path']
    width = data['width']
    height = data['height']
    objects = data['boxes']
    file_name = basename(data['path'])
    file_type = get_file_type(url)
    encoded_image_data = encoded_img

    if not (file_type == 'jpeg' or file_type == 'jpg' or file_type == 'png'):
        raise Exception('Unsupported image file type.')

    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)

    for obj in objects:
        # Using exception here to prevent crashes due to mistakes that may exist in the json file.
        try:
            x0 = obj['x0']
            x1 = obj['x1']
            y0 = obj['y0']
            y1 = obj['y1']
            type = obj['type']
            xmins.append(x0 / width)
            xmaxs.append(x1 / width)
            ymins.append(y0 / height)
            ymaxs.append(y1 / height)
            classes_text.append(type.encode('utf8'))
            classes.append(int(get_id_from_name(type)))
        except KeyError as k:
            print(k.__str__() + "\nSkipping an object in " + url + " because it is missing key(s).")
            continue
            
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(file_name.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(file_name.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(file_type.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main():
    parser = argparse.ArgumentParser(description='Generate a .tfrecord file using the '
                                                 'inputted image files and json data '
                                                 'for deep learning.')
    parser.add_argument('--file_path', default='./metadata.json', help='Path for '
                                                                      'json file.')
    parser.add_argument('--images_path', default='./images', help='Path for the'
                                                                  ' folder '
                                                                  'containing '
                                                                  'the images.')
    args = parser.parse_args()

    file_path = args.file_path
    images_path = args.images_path

    start_conversion(file_path, images_path)


if __name__ == "__main__":
    main()
