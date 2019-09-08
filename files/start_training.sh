#!/bin/bash
cd /tensorflow/models/research
python3 /tensorflow/models/research/object_detection/model_main.py --pipeline_config_path="/pipeline/dl/ssd_mobilenet_v1_coco_2018_01_28/pipeline.config" --model_dir="/pipeline/dl/modeldir" --sample_1_of_n_eval_examples=1 --alsologtostderr & tensorboard --logdir /pipeline/dl/modeldir
