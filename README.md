# E2Pose: Fully Convolutional Networks for End-to-End Multi-Person Pose Estimation

## Abstract

## Demo on Google Colab
[demo_inference.ipynb](http://colab.research.google.com/github/AISIN-TRC/E2Pose/blob/main/demo_inference.ipynb)

## Download pre-train models
```bash
./pretrains/download.sh
```

## Inference E2Pose's demo on localhost
```bash
#inference video
./inference.sh --src './sample/$YOUR_MOVIE.mp4'
#inference image
./inference.sh --src './sample/$YOUR_IMAGE.jpg'
```

## E2Pose's live demo with camera device
### Setup environment @ jetson
    # Add this to your .bashrc file
    export CUDA_HOME=/usr/local/cuda
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export OPENBLAS_CORETYPE=ARMV8
    export PYTHONPATH=/usr/lib/python3.6/dist-packages:$PYTHONPATH

    # Install qt5 libs
    sudo apt-get install \
             qt5-default \
             pyqt5-dev \
             pyqt5-dev-tools \
             python3-pyqt5 \
             qttools5-dev-tools
    
    # Create venv
    python3 -m venv venv

    # Activate venv
    . venv/bin/activate

    # Upgrade pip
    pip3 install --upgrade pip

    # Install pip libs
    pip3 install -r requirements_l4t.txt


### Convert to TensorRT model (Running on Jetson devices takes a very long time)
```bash
./generate_trtmodel.sh
```
#### (Reference) Time required for conversion @ Jetson AGX Xavier
    ResNet101/512x512 : tf2onnx = 108 minutes, onnx2trt = 20 minutes

### Launch the GUI @ docker
```bash
./livedemo.sh
```

### Launch the GUI @ jetson
```bash
. venv/bin/activate
python3 live/demo.py
```

# Benchmark code for comparing framerates
## OpenPifPaf: Composite Fields for Semantic Keypoint Detection and Spatio-Temporal Association [[arxiv](https://arxiv.org/abs/2103.02440)][[github](https://github.com/openpifpaf/openpifpaf)]
### Inference openPifPaf's demo on localhost
```bash
#inference video
./inference_pifpaf.sh --src './sample/$YOUR_MOVIE.mp4'
#inference image
./inference_pifpaf.sh --src './sample/$YOUR_IMAGE.jpg'
```
## OpenMMLab: Pose Estimation Toolbox and Benchmark [[github](https://github.com/open-mmlab/mmpose)]
### inference mmpose's demo on localhsot
```bash
#inference video
./inference_mmpose.sh --src './smaple/$YOUR_MOVIE.mp4'
#inference image
./inference_mmpose.sh --src './sample/$YOUR_IMAGE.jpg'
```


# Commercial License
The open source license is in the [LICENSE](./LICENSE) file. This software is also available for licensing via the AISIN Corp. (https://www.aisin.com/).
