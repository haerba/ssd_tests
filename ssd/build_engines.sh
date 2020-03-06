#!/bin/bash

set -xe

for model in ssd_mobilenet_v1_coco \

             ssd_mobilenet_v2_coco ; do
    python3 build_engine.py ${model}
done
