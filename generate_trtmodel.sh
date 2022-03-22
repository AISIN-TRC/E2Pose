#!/bin/bash
. ./tools/getoptions.sh

VERSION="1.0"
parser_definition() {
    setup   REST help:usage -- "Usage: inference.sh [options]" ''
    msg -- 'Options:'
    param  GPUS       -g --gpus           init:=1                                                    -- "Docker gpus option"
    param  OPSET         --opset          init:=13                                                   -- "ONNX opset"
    param  FREEZED_MODEL --freezed_model  init:='./pretrains/COCO/ResNet101/512x512/frozen_model.pb' -- "[src] Frozen model path"
    param  ONNX_MODEL    --onnx_model     init:='./pretrains/COCO/ResNet101/512x512/model.onnx'      -- "[dst] ONNX model path"
    param  TRT_MODEL     --trt_model      init:='./pretrains/COCO/ResNet101/512x512/model.trt'       -- "[dst] TensorRT model path"
    disp    :usage  -h --help
    disp    VERSION    --version
}
eval "$(getoptions parser_definition) exit 1"
    
if [[ -f /etc/nv_tegra_release ]]; then
    IS_JETSON=true
else
    IS_JETSON=false
fi

if "${IS_JETSON}"; then
    TF_IMG=masakazutobeta/e2pose:nvcr-l4t-r32.6.1-tf2.5-py3.qt5
else
    TF_IMG=masakazutobeta/e2pose:nvcr-21.08-tf2-py3.qt5
fi

docker run --rm --gpus $GPUS --net host -e TF_FORCE_GPU_ALLOW_GROWTH=true\
  -v /etc/localtime:/etc/localtime -v $HOME/.Xauthority:/root/.Xauthority -e DISPLAY=$DISPLAY\
  -v $(pwd):/work -w /work \
  -it $TF_IMG python3 ./live/convert_onnx_trt.py --freezed_model $FREEZED_MODEL --onnx_model $ONNX_MODEL --trt_model $TRT_MODEL