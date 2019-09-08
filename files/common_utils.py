import os
import re
import collections
import json


def get_file_name_without_extension_from_path(src_json_path):
    json_file_name_with_extension = os.path.basename(src_json_path)
    json_file_name_without_extension = os.path.splitext(json_file_name_with_extension)[0]
    return json_file_name_without_extension


def convert_pbtxt_to_json(pbtxt_dir='./files/class_map.pbtxt'):
    with open(pbtxt_dir) as pbtxt:
        pbtxt_str = pbtxt.read()
        id_occurences = [m.start() for m in re.finditer('id', pbtxt_str)]
        name_occurences = [m.start() for m in re.finditer('name', pbtxt_str)]
        items = []

        for id_pos, name_pos in zip(id_occurences, name_occurences):
            id_line_newline_pos = pbtxt_str.find('\n', id_pos)
            name_line_newline_pos = pbtxt_str.find('\n', name_pos)
            id_line_str = pbtxt_str[id_pos: id_line_newline_pos]
            name_line_str = pbtxt_str[name_pos: name_line_newline_pos]
            id = [str(c) for c in id_line_str if c.isdigit()]
            name = re.findall("'([^']*)'", name_line_str)[0]
            item = (("id", ''.join(id)), ("name", name))
            item = collections.OrderedDict(item)
            items.append(item)

        with open('./files/class_map.json', 'w') as outfile:
            json.dump(items, outfile, indent=4)


def get_name_from_id(id, json_dir='./files/class_map.json'):
    if os.path.isfile(json_dir):
        with open(json_dir) as json_file:
            data = json.load(json_file)
            for item in data:
                if item['id'] == str(id):
                    return item['name']


def get_id_from_name(name, json_dir='./files/class_map.json'):
    if os.path.isfile(json_dir):
        with open(json_dir) as json_file:
            data = json.load(json_file)
            for item in data:
                if item['name'] == name:
                    return item['id']


def get_num_of_classes(json_dir='./files/class_map.json'):
    if os.path.isfile(json_dir):
        with open(json_dir) as json_file:
            data = json.load(json_file)
            return len(data)

