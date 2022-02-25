#!/bin/bash
. ./tools/getoptions.sh

VERSION="1.0"
parser_definition() {
    setup   REST help:usage -- "Usage: inference_mmpose.sh [options]" ''
    msg -- 'Options:'
    param  GPUS     -g --gpus           init:="1"                                            -- "Docker gpus option"
    param  SRC         --src            init:='./sample/*'                                   -- "Path of the input image/movies. You can also specify using asterisks."
    param  DST_DIR     --dst            init:='./sample_out/mmpose'                           -- "Output directory"
    param  MMDET       --det            init:='yolov3_d53_320_273e_coco'                     -- "Object detection model name"
    param  MMPOSE      --pose           init:='shufflenetv2_coco_256x192'                    -- "Pose estimation model name"
    param  TORCH_HOME  --torch_home     init:='./.mmpose/torch_home'                         -- "Torch home directory"
    flag   OUTPUT_DEV  --dev            on:true of:false  init:=false                        -- "Output information for development"
    flag   ADD_FPS     --off_fps        on:false of:true  init:=true                         -- "Disable the FPS text display"
    flag   ADD_BLUR    --add_blur       on:true of:false  init:=false                        -- "Add a blur mosaic"
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
    TF_IMG=masakazutobeta/mmpose:inference-nvcr-l4t-r32.6.1-pth1.8-py3
else
    TF_IMG=masakazutobeta/mmpose:inference-nvcr-20.11-py3
fi

docker run --rm --gpus $GPUS --net host --shm-size=4g\
  -v /etc/localtime:/etc/localtime -v $HOME/.Xauthority:/root/.Xauthority\
  -e DISPLAY=$DISPLAY\
  -e TORCH_HOME=$TORCH_HOME\
  -e CUDA_LAUNCH_BLOCKING=1\
  -v $(pwd):/work -w /work\
  -it $TF_IMG python3 inference_mmpose.py --src "$SRC" --dst $DST_DIR --det $MMDET --pose $MMPOSE --dev $OUTPUT_DEV --add_fps $ADD_FPS --add_blur $ADD_BLUR