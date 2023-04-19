# E2Pose: Fully Convolutional Networks for End-to-End Multi-Person Pose Estimation

## Abstract
Highly accurate multi-person pose estimation at a high framerate is a fundamental problem in autonomous driving. Solving the problem could aid in preventing pedestrian-car accidents. The present study tackles this problem by proposing a new model composed of a feature pyramid and an original head to a general backbone. The original head is built using lightweight CNNs and directly estimates multi-person pose coordinates. This configuration avoids the complex post-processing and two-stage estimation adopted by other models and allows for a lightweight model. Our model can be trained end-to-end and performed in real-time on a resource-limited platform (low-cost edge device) during inference. Experimental results using the COCO and CrowdPose datasets showed that our model can achieve a higher framerate (approx. 20 frames/sec with NVIDIA Jetson AGX Xavier) than other state-of-the-art models while maintaining sufficient accuracy for practical use.

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

## Convert to TensorRT model (Running on Jetson devices takes a very long time)
```bash
./generate_trtmodel.sh
```
### (Reference) Time required for conversion @ Jetson AGX Xavier
    ResNet101/512x512 : tf2onnx = 108 minutes, onnx2trt = 20 minutes

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

# Citation
Masakazu Tobeta, Yoshihide Sawada, Ze Zheng, Sawa Takamuku, Naotake Natori. "[E2Pose: Fully Convolutional Networks for End-to-End Multi-Person Pose Estimation](https://ieeexplore.ieee.org/document/9981322)". 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS).

戸部田雅一，鄭澤，高椋佐和，澤田好秀， "[高精度及び高フレームレートなEnd‐to‐End多人数姿勢推定](https://www.aisin.com/jp/technology/technicalreview/27/pdf/08.pdf)"．アイシン技報2023．

# Commercial License
The open source license is in the [LICENSE](./LICENSE) file. This software is also available for licensing via the AISIN Corp. (https://www.aisin.com/).
