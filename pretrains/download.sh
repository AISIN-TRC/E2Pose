#!/bin/bash
mkdir -p ./COCO/ResNet101/512x512/saved_model
wget -v -O ./COCO/ResNet101/512x512/frozen_model.pb -L https://ent.box.com/shared/static/hqsrequ6e672gayggoh19qppri294th9.pb
