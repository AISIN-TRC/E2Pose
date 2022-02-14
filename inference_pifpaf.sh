#!/bin/bash
. ./tools/getoptions.sh

VERSION="1.0"
parser_definition() {
    setup   REST help:usage -- "Usage: inference_pifpaf.sh [options]" ''
    msg -- 'Options:'
    param  GPUS     -g --gpus           init:="1"                                            -- "Docker gpus option"
    param  SRC         --src            init:='./sample/*'                                   -- "Path of the input image/movies. You can also specify using asterisks."
    param  DST_DIR     --dst            init:='./sample_out/pifpaf'                           -- "Output directory"
    param  CKPT        --ckpt           init:='shufflenetv2k16'                              -- "Checkpoint name"
    param  TORCH_HOME  --torch_home     init:='./.pifpaf/torch_home'                         -- "Torch home directory"
    param  INPUT_HW    --input_wh       init:='512,512'                                      -- "Input image size to the model"
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
    TF_IMG=masakazutobeta/openpifpaf:nvcr-l4t-r32.6.1-pth1.8-py3
else
    TF_IMG=masakazutobeta/openpifpaf:nvcr-21.06-py3
fi

docker run --rm --gpus $GPUS --net host --shm-size=4g\
  -v /etc/localtime:/etc/localtime -v $HOME/.Xauthority:/root/.Xauthority\
  -e DISPLAY=$DISPLAY\
  -e TORCH_HOME=$TORCH_HOME\
  -v $(pwd):/work -w /work\
  -it $TF_IMG python3 inference_pifpaf.py --src "$SRC" --dst $DST_DIR --ckpt $CKPT --dev $OUTPUT_DEV --add_fps $ADD_FPS --add_blur $ADD_BLUR --input_wh $INPUT_HW