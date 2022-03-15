#!/bin/bash
. ./tools/getoptions.sh

VERSION="1.0"
parser_definition() {
    setup   REST help:usage -- "Usage: livedemo.sh [options]" ''
    msg -- 'Options:'
    param   VIDEO   -v --video     init:="./sample/2022-02-08-22-24-58.mp4"
    disp    :usage  -h --help
    disp    VERSION    --version
}
eval "$(getoptions parser_definition) exit 1"

# Reference: https://www.linuxfordevices.com/tutorials/linux/fake-webcam-streams
# sudo apt install v4l2loopback-dkms
# sudo apt install ffmpeg

sudo modprobe --remove v4l2loopback
sudo modprobe v4l2loopback card_label="My Fake Webcam" exclusive_caps=1 video_nr=99
sudo chmod 777 /dev/video99
ffmpeg -stream_loop -1 -re -i $VIDEO -vcodec rawvideo -threads 0 -f v4l2 /dev/video99
sudo modprobe --remove v4l2loopback