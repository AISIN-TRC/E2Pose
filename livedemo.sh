#!/bin/bash
. ./tools/getoptions.sh

VERSION="1.0"
parser_definition() {
    setup   REST help:usage -- "Usage: livedemo.sh [options]" ''
    msg -- 'Options:'
    param   GPUS    -g --gpus     init:="all"           -- "Docker gpus option"
    param   CAMERA  -c --camera   init:="/dev/video99"  -- "Camera device path"
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
  -v /etc/localtime:/etc/localtime -v $HOME/.Xauthority:/root/.Xauthority -e DISPLAY=$DISPLAY --device $CAMERA:/dev/video0:mwr \
  -v $(pwd):/work -w /work\
  -it $TF_IMG python3 ./live/demo.py