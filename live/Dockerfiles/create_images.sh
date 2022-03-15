docker build --build-arg http_proxy=$HTTP_PROXY --build-arg https_proxy=$HTTP_PROXY --tag masakazutobeta/e2pose:nvcr-21.06-tf2-py3.qt5 -f ./E2Pose .
echo 'COMPLEAT!'