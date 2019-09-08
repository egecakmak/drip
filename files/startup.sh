#!/bin/bash
export PYTHONPATH=:/tensorflow/models/research:/tensorflow/models/research/slim
cd /pipeline/files
cp -f json_to_tfrecord.py preprocess_images.py pipeline_runner.py common_utils.py  ..
cd /pipeline
python3 pipeline_runner.py --eval_percentage 20 --batch_size 4 --number_of_steps 200000