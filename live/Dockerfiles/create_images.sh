#!/bin/bash
if [[ -f /etc/nv_tegra_release ]]; then
    IS_JETSON=true
else
    IS_JETSON=false
fi

if "${IS_JETSON}"; then
    docker build --build-arg http_proxy=$HTTP_PROXY --build-arg https_proxy=$HTTP_PROXY --tag masakazutobeta/e2pose:nvcr-l4t-r32.6.1-tf2.5-py3.qt5 -f ./E2PoseL4T .
else
    docker build --build-arg http_proxy=$HTTP_PROXY --build-arg https_proxy=$HTTP_PROXY --tag masakazutobeta/e2pose:nvcr-21.08-tf2-py3.qt5 -f ./E2Pose .
fi

echo 'COMPLEAT!'