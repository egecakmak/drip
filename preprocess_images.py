import cv2
import argparse
import json
import random

from common_utils import get_id_from_name, get_name_from_id, get_file_name_without_extension_from_path
from collections import OrderedDict
from os import path, makedirs, getcwd
from albumentations import HorizontalFlip, RandomRotate90, Transpose, VerticalFlip, RandomCrop, Resize, \
    ElasticTransform, \
    RandomFog, RandomRain, JpegCompression, GaussianBlur, GaussNoise, RGBShift, RandomGamma, RandomSnow, RandomSunFlare, \
    HueSaturationValue, OpticalDistortion, MotionBlur, RandomShadow, Compose, OneOf, OneOrOther
from pathlib import Path


def start_preprocessing(src_json_path, src_img_path, draw_borders):
    print('Starting augmenting images... \n')
    augmented_data = []
    with open(src_json_path) as json_file:
        data = json.load(json_file, object_pairs_hook=OrderedDict)  # Loads the json file into an ordered dictionary.
        length = len(data['frames'])
        for num, frame in enumerate(data['frames'], start=1):
            annotations = annotate_augmentations(frame, src_img_path)
            augmented = get_augmented_image(annotations)
            save_image(augmented['image'], frame['path'], augmented['bboxes'], draw_borders)
            del augmented['image']
            augmented_data.append(augmented)
            percent_completed = int(num / length * 100)
            if percent_completed % 1 == 0 or percent_completed == 0:
                print(str(percent_completed) + "% completed.", end='\r')

        json_file_name_without_extension = get_file_name_without_extension_from_path(src_json_path)
        new_json_file_name = json_file_name_without_extension + '_augmented.json'
        save_json(augmented_data, new_json_file_name)

    print('******************************************************')
    print('Done augmenting images. \n')


def read_image(path):
    # print('Reading file: ' + path)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converts the image colors from BGR to RGB.
    return image


def save_image(img, file_path, objects, draw_borders):
    if draw_borders:  # Draws borders around objects detected if parameter is True.
        for obj in objects:
            x0 = int(obj[0])
            y0 = int(obj[1])
            x1 = int(obj[2])
            y1 = int(obj[3])
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 3)

    file_name = path.basename(file_path)
    new_file_path = './images_augmented/' + file_path
    parent_path = new_file_path.replace(file_name, '')
    new_name = get_file_name_without_extension_from_path(file_name) + '_augmented' + Path(new_file_path).suffix
    new_file_path = new_file_path.replace(file_name, new_name)

    # print('Creating file: ' + new_file_path + '\n')

    # Creates the directories where the images will be stored.
    if not path.exists(parent_path):
        makedirs(parent_path)
    cv2.imwrite(new_file_path, img)


def save_json(augmented, filename):
    print('Creating file: ' + filename)
    frames = []

    for frame in augmented:
        for bbox, category_id in zip(frame['bboxes'], frame['category_id']):
            objs = []
            y = (("type", get_name_from_id(category_id)), ("x0", bbox[0]), ("y0", bbox[1]), ("x1", bbox[2]),
                 ("y1", bbox[3]))
            y = OrderedDict(y)
            objs.append(y)

        file_path = frame['frame_url']
        file_name = path.basename(frame['frame_url'])
        new_name = get_file_name_without_extension_from_path(file_name) + '_augmented' + Path(file_path).suffix
        file_path = file_path.replace(file_name, new_name)
        x = (("path", file_path), ("width", frame['image_width']),
             ("height", frame['image_height']), ("boxes", objs))
        x = OrderedDict(x)
        frames.append(x)

    z = (("frames", frames),)
    z = OrderedDict(z)

    with open(filename, 'w') as outfile:
        json.dump(z, outfile, indent=4)


def annotate_augmentations(frame, img_path):
    # print('Annotating: ' + frame['frame_url'])
    image_path = frame['path']
    image_width = frame['width']
    image_height = frame['height']
    image_objects = frame['boxes']

    image = read_image(img_path + image_path)
    annotations = {'image': image, 'bboxes': [], 'category_id': [], 'frame_url': image_path,
                   'image_width': image_width, 'image_height': image_height}

    for obj in image_objects:
        # Using exceptions here to prevent crashes due to mistakes that may exist in the json file.
        try:
            obj_type = obj['type']
            x0 = obj['x0']
            y0 = obj['y0']
            x1 = obj['x1']
            y1 = obj['y1']
        except KeyError as k:
            print(k.__str__() + "\nSkipping an object in " + frame['path'] + " because it is missing key(s).")
            continue
        obj_class_id = int(get_id_from_name(obj_type))
        annotations['bboxes'].append([x0, y0, x1, y1])
        annotations['category_id'].append(obj_class_id)

    return annotations


def get_random_aug(min_area=0., min_visibility=0.):
    aug = Compose([RandomRotate90(p=random.uniform(0, 1)), GaussianBlur(p=random.uniform(0, 1)),
                   GaussNoise(p=random.uniform(0, 1)),
                   HueSaturationValue(p=random.uniform(0, 1)), RGBShift(p=random.uniform(0, 1))])
    return Compose(aug, bbox_params={'format': 'pascal_voc',
                                     'min_area': min_area,
                                     'min_visibility': min_visibility,
                                     'label_fields': ['category_id']})


def get_augmented_image(annotations):
    # print('Augmenting: ' + annotations['frame_url'])
    aug = get_random_aug(min_visibility=0.3)
    # Below loop checks for exceptions that may be raised due to incorrect bbox coordinates and cleans those bboxes
    # with incorrect coordinates and then tries again.
    while True:
        try:
            aug = aug(**annotations)
        except ValueError as v:
            bboxes_before_deletion = len(annotations['bboxes'])
            annotations['bboxes'] = [bb for bb in annotations['bboxes'] if bb[0] < bb[2] and bb[1] < bb[3]]
            bboxes_after_deletion = len(annotations['bboxes'])
            num_of_boxes_deleted = bboxes_before_deletion - bboxes_after_deletion
            print(v.__str__() + "\n Skipping " + str(num_of_boxes_deleted) +
                  " object(s) in " + annotations['frame_url'] + " due to incorrect  bbox coordinates.")
            continue
        break
    return aug


def main():
    parser = argparse.ArgumentParser(description='Augment images randomly.')

    parser.add_argument('--src_json_path', default='./metadata.json', help='Path for the original json file.')
    parser.add_argument('--src_img_path', default='./images/', help='Path for the folder containing images.')

    parser.add_argument('--draw_obj_borders', action='store_true', help='Draw borders around detected objects.')
    args = parser.parse_args()
    src_json_path = args.src_json_path
    src_img_path = args.src_img_path
    draw_borders = args.draw_obj_borders

    start_preprocessing(src_json_path, src_img_path, draw_borders)


if __name__ == '__main__':
    main()
