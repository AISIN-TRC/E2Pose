#!/bin/bash
. ./tools/getoptions.sh

VERSION="1.0"
parser_definition() {
    setup   REST help:usage -- "Usage: inference.sh [options]" ''
    msg -- 'Options:'
    param  GPUS     -g --gpus      init:="all"                                            -- "Docker gpus option"
    param  SRC         --src       init:='./sample/*'                                     -- "Path of the input image/movies. You can also specify using asterisks."
    param  DST_DIR     --dst       init:='./sample_out/e2pose'                            -- "Output directory"
    param  MODEL       --model     init:='./pretrains/COCO/ResNet101/512x512/saved_model' -- "Model path"
    param  BACKBONE    --backbone  init:='ResNet101'                                      -- "Name of the Backbone"
    flag   OUTPUT_DEV  --dev       on:true of:false  init:=false                          -- "Output information for development"
    flag   ADD_FPS     --off_fps   on:false of:true  init:=true                           -- "Disable the FPS text display"
    flag   ADD_BLUR    --add_blur  on:true of:false  init:=false                          -- "Add a blur mosaic"
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
    TF_IMG=masakazutobeta/e2pose:nvcr-l4t-r32.6.1-tf2.5-py3
else
    TF_IMG=masakazutobeta/e2pose:nvcr-21.06-tf2-py3.v1
fi

docker run --rm --gpus $GPUS --net host -e TF_FORCE_GPU_ALLOW_GROWTH=true\
  -v /etc/localtime:/etc/localtime -v $HOME/.Xauthority:/root/.Xauthority -e DISPLAY=$DISPLAY\
  -v $(pwd):/work -w /work\
  -it $TF_IMG python3 inference.py --src "$SRC" --dst $DST_DIR --model $MODEL --backbone $BACKBONE --dev $OUTPUT_DEV --add_fps $ADD_FPS --add_blur $ADD_BLUR
  