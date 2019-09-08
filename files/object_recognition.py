import argparse
import os
import cv2
import numpy as np
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import json

from distutils.version import StrictVersion
from collections import defaultdict, OrderedDict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util
from common_utils import get_id_from_name, get_name_from_id, get_file_name_without_extension_from_path


def process_path(path, graph, label_map, results_path, threshold):
    images = []
    for image_name in os.listdir(path):
        print('Preparing ' + image_name + ' for inference.')
        image_path = path + '/' + image_name
        if os.path.isfile(image_path):
            image = read_image(image_path)
            try:
                image_array = load_image_into_numpy_array(image)
            except OSError as e:
                print('Skipping image ' + image_name + ' due to an OSError.')
                print(e.__str__())
                continue
            image_array_expanded = np.expand_dims(image_array, axis=0)  # Expanding the image array so that we can infer.
            image_obj = {
                "image_name": image_name,
                "image": image_array_expanded
            }
            images.append(image_obj)

    detection_graph = load_model(graph)
    category_index = load_label_map(label_map)

    output_dicts = recognize_objects(images, detection_graph)
    path_basename = os.path.basename(path)

    new_json_file_name = path_basename + '.json'
    save_json(output_dicts, new_json_file_name, results_path, threshold)


def read_image(path):
    return Image.open(path)


def load_model(graph):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def load_label_map(label_map):
    return label_map_util.create_category_index_from_labelmap(label_map, use_display_name=True)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def recognize_objects(image_objs, model):
    with model.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)

            output_dicts = []

            for image_obj in image_objs:
                if 'detection_masks' in tensor_dict:
                    # The following process is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image_obj['image'].shape[1],
                        image_obj['image'].shape[2])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)

                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                image = image_obj['image']
                name = image_obj['image_name']

                print('Inferring objects in ' + name + '.')
                # Run inference
                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image})
                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['image_height'] = image_obj['image'].shape[1]
                output_dict['image_width'] = image_obj['image'].shape[2]
                output_dict['image_name'] = name
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.int64)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
                output_dicts.append(output_dict)

    return output_dicts


def save_json(output_dicts, new_json_file_name, results_path, threshold):
    results = []
    for output_dict in output_dicts:
        objeler = []
        image_name = output_dict['image_name']
        image_id = extract_image_id(image_name)
        image_height = output_dict['image_height']
        image_width = output_dict['image_width']
        for i in range(output_dict['num_detections']):
            det_score = output_dict['detection_scores'][i] * 100
            if det_score > threshold:
                det_class = output_dict['detection_classes'][i]
                det_boxes = output_dict['detection_boxes'][i]
                y1 = round(det_boxes[0] * image_height)
                x1 = round(det_boxes[1] * image_width)
                y2 = round(det_boxes[2] * image_height)
                x2 = round(det_boxes[3] * image_width)
                obj = (("type", get_name_from_id(int(det_class))), ("x1", x1), ("y1", y1), ("x2", x2),
                       ("y2", y2))
                obj = OrderedDict(obj)
                objeler.append(obj)

        y = (("id", image_id), ("objects", objeler))
        y = OrderedDict(y)
        results.append(y)

    if not os.path.isdir('./files/results'):
        os.makedirs('./files/results')
        
    # original_path = os.getcwd()
    # os.chdir(results_path)
    with open('./files/results/' + new_json_file_name, 'w') as outfile:
        print('Saving result file: ' + new_json_file_name)
        json.dump(results, outfile, indent=4)
    # os.chdir(original_path)


def extract_image_id(image_name):
    result_str = ''
    for ch in image_name:
        if ch.isdigit():
            result_str = result_str + ch
    return int(result_str)


def main():
    Image.LOAD_TRUNCATED_IMAGES = True
    parser = argparse.ArgumentParser(description='Recognize objects in a given set of images using the model trained.')

    parser.add_argument('--graph', default='./files/graph.pb', help='Path for the trained graph.')

    parser.add_argument('--images_path', default='/images', help='Path for the folder containing image folders.')

    parser.add_argument('--label_map_path', default='./files/class_map.pbtxt', help='Path for the label map file.')

    parser.add_argument('--results_path', default='./', help='Path where the resulting json files will be saved.')

    parser.add_argument('--threshold', default=60, help='Threshold.')

    args = parser.parse_args()

    graph = args.graph
    images_path = args.images_path
    label_map_path = args.label_map_path
    results_path = args.results_path
    threshold = int(args.threshold)

    for folder in os.listdir(images_path):
        image_folder_path = images_path + '/' + folder
        if os.path.isdir(image_folder_path):
            print('Processing folder: ' + image_folder_path )
            process_path(image_folder_path, graph, label_map_path, results_path, threshold)


if __name__ == '__main__':
    main()
