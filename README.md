# E2Pose
End-to-end lightweight pose estimation method

## Abstract

## Demo on Google Colab
[demo_inference.ipynb](http://colab.research.google.com/github/AISIN-TRC/E2Pose/blob/main/demo_inference.ipynb)


## Download inference models
```bash
cd ./pretrains
./download.sh
```

## Inference E2Pose's demo on localhost
```bash
#inference video
./inference.sh --src ./sample/$YOUR_MOVIE.mp4
#inference image
./inference.sh --src ./sample/$YOUR_IMAGE.jpg
```

## Inference [openPifPaf](https://openpifpaf.github.io/intro.html)'s demo on localhost
```bash
#inference video
./inference_pifpaf.sh --src ./sample/$YOUR_MOVIE.mp4
#inference image
./inference_pifpaf.sh --src ./sample/$YOUR_IMAGE.jpg
```

## inference [mmpose](https://mmpose.readthedocs.io/en/latest/)'s demo on localhsot
```bash
#inference video
./inference_mmpose.sh --src ./smaple/$YOUR_MOVIE.mp4
#inference image
./inference_mmpose.sh --src ./sample/$YOUR_IMAGE.jpg
```


## Commercial License
The open source license is in the [LICENSE](./LICENSE) file. This software is also available for licensing via the AISIN Corp. (https://www.aisin.com/).
