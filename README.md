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
