# --- Reference ---
# https://mmdetection.readthedocs.io/en/latest/model_zoo.html
# https://mmpose.readthedocs.io/en/latest/topics/body%282d%2Ckpt%2Csview%2Cimg%29.html

# --- Defines ---
DETECTORS = [
['/mmpose/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_coco.py', 'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1x_coco/cascade_rcnn_x101_64x4d_fpn_1x_coco_20200515_075702-43ce6a30.pth'],
['/mmpose/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py', 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'],
['/mmpose/mmdetection_cfg/ssdlite_mobilenetv2_scratch_600e_coco.py', 'https://download.openmmlab.com/mmdetection/v2.0/ssd/ssdlite_mobilenetv2_scratch_600e_coco/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth'],
['/mmpose/mmdetection_cfg/yolov3_d53_320_273e_coco.py', 'https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_320_273e_coco/yolov3_d53_320_273e_coco-421362b6.pth'],
]

BOTTOMUP_POSEMODELS = [
["/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/mobilenetv2_coco_512x512.py",
 "https://download.openmmlab.com/mmpose/bottom_up/mobilenetv2_coco_512x512-4d96e309_20200816.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/mobilenetv2_coco_512x512.py",
 "https://download.openmmlab.com/mmpose/bottom_up/mobilenetv2_coco_512x512-4d96e309_20200816.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/res50_coco_512x512.py",
 "https://download.openmmlab.com/mmpose/bottom_up/res50_coco_512x512-5521bead_20200816.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/res50_coco_640x640.py",
 "https://download.openmmlab.com/mmpose/bottom_up/res50_coco_640x640-2046f9cb_20200822.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/res101_coco_512x512.py",
 "https://download.openmmlab.com/mmpose/bottom_up/res101_coco_512x512-e0c95157_20200816.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/res152_coco_512x512.py",
 "https://download.openmmlab.com/mmpose/bottom_up/res152_coco_512x512-364eb38d_20200822.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/res50_coco_512x512.py",
 "https://download.openmmlab.com/mmpose/bottom_up/res50_coco_512x512-5521bead_20200816.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/res50_coco_640x640.py",
 "https://download.openmmlab.com/mmpose/bottom_up/res50_coco_640x640-2046f9cb_20200822.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/res101_coco_512x512.py",
 "https://download.openmmlab.com/mmpose/bottom_up/res101_coco_512x512-e0c95157_20200816.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/res152_coco_512x512.py",
 "https://download.openmmlab.com/mmpose/bottom_up/res152_coco_512x512-364eb38d_20200822.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w32_coco_512x512_udp.py",
 "https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_coco_512x512_udp-8cc64794_20210222.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w48_coco_512x512_udp.py",
 "https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet48_coco_512x512_udp-7cad61ef_20210222.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w32_coco_512x512.py",
 "https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_512x512-bcb8c247_20200816.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w48_coco_512x512.py",
 "https://download.openmmlab.com/mmpose/bottom_up/hrnet_w48_coco_512x512-cf72fcdf_20200816.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w32_coco_512x512.py",
 "https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_512x512-bcb8c247_20200816.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w48_coco_512x512.py",
 "https://download.openmmlab.com/mmpose/bottom_up/hrnet_w48_coco_512x512-cf72fcdf_20200816.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w32_coco_512x512.py",
 "https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_coco_512x512-8ae85183_20200713.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w32_coco_640x640.py",
 "https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_coco_640x640-a22fe938_20200712.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w48_coco_512x512.py",
 "https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet48_coco_512x512-60fedcbc_20200712.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w32_coco_512x512.py",
 "https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_coco_512x512-8ae85183_20200713.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w32_coco_640x640.py",
 "https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_coco_640x640-a22fe938_20200712.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w48_coco_512x512.py",
 "https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet48_coco_512x512-60fedcbc_20200712.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w32_coco_512x512_udp.py",
 "https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_512x512_udp-91663bf9_20210220.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w48_coco_512x512_udp.py",
 "https://download.openmmlab.com/mmpose/bottom_up/hrnet_w48_coco_512x512_udp-de08fd8c_20210222.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hourglass_ae_coco_512x512.py",
 "https://download.openmmlab.com/mmpose/bottom_up/hourglass_ae/hourglass_ae_coco_512x512-90af499f_20210920.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hourglass_ae_coco_512x512.py",
 "https://download.openmmlab.com/mmpose/bottom_up/hourglass_ae/hourglass_ae_coco_512x512-90af499f_20210920.pth"],
]
BOTTOMUP_POSEMODELS = [','.join(item) for item in BOTTOMUP_POSEMODELS]
BOTTOMUP_POSEMODELS = list(set(BOTTOMUP_POSEMODELS))
BOTTOMUP_POSEMODELS = [item.split(',') for item in BOTTOMUP_POSEMODELS]

TOPDOWN_POSEMODELS = [
["/mmpose/configs/body/2d_kpt_sview_rgb_img/deeppose/coco/res50_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res50_coco_256x192-f6de6c0e_20210205.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/deeppose/coco/res101_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res101_coco_256x192-2f247111_20210205.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/deeppose/coco/res152_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res152_coco_256x192-7df89a88_20210205.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/shufflenetv2_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/shufflenetv2/shufflenetv2_coco_256x192-0aba71c7_20200921.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/shufflenetv2_coco_384x288.py",
 "https://download.openmmlab.com/mmpose/top_down/shufflenetv2/shufflenetv2_coco_384x288-fb38ac3a_20200921.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/litehrnet_18_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/litehrnet/litehrnet18_coco_256x192-6bace359_20211230.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/litehrnet_18_coco_384x288.py",
 "https://download.openmmlab.com/mmpose/top_down/litehrnet/litehrnet18_coco_384x288-8d4dac48_20211230.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/litehrnet_30_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/litehrnet/litehrnet30_coco_256x192-4176555b_20210626.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/litehrnet_30_coco_384x288.py",
 "https://download.openmmlab.com/mmpose/top_down/litehrnet/litehrnet30_coco_384x288-a3aef5c4_20210626.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hourglass52_coco_256x256.py",
 "https://download.openmmlab.com/mmpose/top_down/hourglass/hourglass52_coco_256x256-4ec713ba_20200709.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hourglass52_coco_384x384.py",
 "https://download.openmmlab.com/mmpose/top_down/hourglass/hourglass52_coco_384x384-be91ba2b_20200812.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192_coarsedropout.py",
 "https://download.openmmlab.com/mmpose/top_down/augmentation/hrnet_w32_coco_256x192_coarsedropout-0f16a0ce_20210320.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192_gridmask.py",
 "https://download.openmmlab.com/mmpose/top_down/augmentation/hrnet_w32_coco_256x192_gridmask-868180df_20210320.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192_photometric.py",
 "https://download.openmmlab.com/mmpose/top_down/augmentation/hrnet_w32_coco_256x192_photometric-308cf591_20210320.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_256x192_fp16_dynamic.py",
 "https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192_fp16_dynamic-6edb79f3_20210430.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/mobilenetv2_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/mobilenetv2/mobilenetv2_coco_256x192-d1e58e7b_20200727.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/mobilenetv2_coco_384x288.py",
 "https://download.openmmlab.com/mmpose/top_down/mobilenetv2/mobilenetv2_coco_384x288-26be4816_20200727.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_384x288.py",
 "https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_384x288-e6f795e9_20200709.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res101_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_256x192-6e6babf0_20200708.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res101_coco_384x288.py",
 "https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_384x288-8c71bdc9_20200709.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res152_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_256x192-f6e307c2_20200709.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res152_coco_384x288.py",
 "https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_384x288-3860d4c9_20200709.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_384x288.py",
 "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_384x288-d9f0d786_20200708.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_384x288.py",
 "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288-314c8528_20200708.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/rsn18_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/rsn/rsn18_coco_256x192-72f4b4a7_20201127.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/rsn50_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/rsn/rsn50_coco_256x192-72ffe709_20201127.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/2xrsn50_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/rsn/2xrsn50_coco_256x192-50648f0e_20201127.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/3xrsn50_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/rsn/3xrsn50_coco_256x192-58f57a68_20201127.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnest50_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/resnest/resnest50_coco_256x192-6e65eece_20210320.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnest50_coco_384x288.py",
 "https://download.openmmlab.com/mmpose/top_down/resnest/resnest50_coco_384x288-dcd20436_20210320.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnest101_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/resnest/resnest101_coco_256x192-2ffcdc9d_20210320.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnest101_coco_384x288.py",
 "https://download.openmmlab.com/mmpose/top_down/resnest/resnest101_coco_384x288-80660658_20210320.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnest200_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/resnest/resnest200_coco_256x192-db007a48_20210517.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnest200_coco_384x288.py",
 "https://download.openmmlab.com/mmpose/top_down/resnest/resnest200_coco_384x288-b5bb76cb_20210517.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnest269_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/resnest/resnest269_coco_256x192-2a7882ac_20210517.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnest269_coco_384x288.py",
 "https://download.openmmlab.com/mmpose/top_down/resnest/resnest269_coco_384x288-b142b9fb_20210517.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnext50_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/resnext/resnext50_coco_256x192-dcff15f6_20200727.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnext50_coco_384x288.py",
 "https://download.openmmlab.com/mmpose/top_down/resnext/resnext50_coco_384x288-412c848f_20200727.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnext101_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/resnext/resnext101_coco_256x192-c7eba365_20200727.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnext101_coco_384x288.py",
 "https://download.openmmlab.com/mmpose/top_down/resnext/resnext101_coco_384x288-f5eabcd6_20200727.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnext152_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/resnext/resnext152_coco_256x192-102449aa_20200727.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnext152_coco_384x288.py",
 "https://download.openmmlab.com/mmpose/top_down/resnext/resnext152_coco_384x288-806176df_20200727.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/mspn50_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/mspn/mspn50_coco_256x192-8fbfb5d0_20201123.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/2xmspn50_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/mspn/2xmspn50_coco_256x192-c8765a5c_20201123.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/3xmspn50_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/mspn/3xmspn50_coco_256x192-e348f18e_20201123.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/4xmspn50_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/mspn/4xmspn50_coco_256x192-7b837afb_20201123.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vgg16_bn_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/vgg/vgg16_bn_coco_256x192-7e7c58d6_20210517.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_256x192_dark.py",
 "https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192_dark-43379d20_20200709.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_384x288_dark.py",
 "https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_384x288_dark-33d3e5e5_20210203.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res101_coco_256x192_dark.py",
 "https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_256x192_dark-64d433e6_20200812.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res101_coco_384x288_dark.py",
 "https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_384x288_dark-cb45c88d_20210203.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res152_coco_256x192_dark.py",
 "https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_256x192_dark-ab4840d5_20200812.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res152_coco_384x288_dark.py",
 "https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_384x288_dark-d3b8ebd7_20210203.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192_udp.py",
 "https://download.openmmlab.com/mmpose/top_down/udp/hrnet_w32_coco_256x192_udp-aba0be42_20210220.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_384x288_udp.py",
 "https://download.openmmlab.com/mmpose/top_down/udp/hrnet_w32_coco_384x288_udp-e97c1a0f_20210223.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192_udp.py",
 "https://download.openmmlab.com/mmpose/top_down/udp/hrnet_w48_coco_256x192_udp-2554c524_20210223.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_384x288_udp.py",
 "https://download.openmmlab.com/mmpose/top_down/udp/hrnet_w48_coco_384x288_udp-0f89c63e_20210223.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192_udp_regress.py",
 "https://download.openmmlab.com/mmpose/top_down/udp/hrnet_w32_coco_256x192_udp_regress-be2dbba4_20210222.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/alexnet_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/alexnet/alexnet_coco_256x192-a7b1fd15_20200727.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/seresnet50_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/seresnet/seresnet50_coco_256x192-25058b66_20200727.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/seresnet50_coco_384x288.py",
 "https://download.openmmlab.com/mmpose/top_down/seresnet/seresnet50_coco_384x288-bc0b7680_20200727.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/seresnet101_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/seresnet/seresnet101_coco_256x192-83f29c4d_20200727.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/seresnet101_coco_384x288.py",
 "https://download.openmmlab.com/mmpose/top_down/seresnet/seresnet101_coco_384x288-48de1709_20200727.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/seresnet152_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/seresnet/seresnet152_coco_256x192-1c628d79_20200727.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/seresnet152_coco_384x288.py",
 "https://download.openmmlab.com/mmpose/top_down/seresnet/seresnet152_coco_384x288-58b23ee8_20200727.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/shufflenetv1_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/shufflenetv1/shufflenetv1_coco_256x192-353bc02c_20200727.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/shufflenetv1_coco_384x288.py",
 "https://download.openmmlab.com/mmpose/top_down/shufflenetv1/shufflenetv1_coco_384x288-b2930b24_20200804.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/cpm_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/cpm/cpm_coco_256x192-aa4ba095_20200817.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/cpm_coco_384x288.py",
 "https://download.openmmlab.com/mmpose/top_down/cpm/cpm_coco_384x288-80feb4bc_20200821.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192_dark.py",
 "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192_dark-07f147eb_20200812.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_384x288_dark.py",
 "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_384x288_dark-307dafc2_20210203.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192_dark.py",
 "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192_dark-8cba3197_20200812.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_384x288_dark.py",
 "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288_dark-e881a4b6_20210203.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/scnet50_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/scnet/scnet50_coco_256x192-6920f829_20200709.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/scnet50_coco_384x288.py",
 "https://download.openmmlab.com/mmpose/top_down/scnet/scnet50_coco_384x288-9cacd0ea_20200709.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/scnet101_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/scnet/scnet101_coco_256x192-6d348ef9_20200709.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/scnet101_coco_384x288.py",
 "https://download.openmmlab.com/mmpose/top_down/scnet/scnet101_coco_384x288-0b6e631b_20200709.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnetv1d50_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d50_coco_256x192-a243b840_20200727.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnetv1d50_coco_384x288.py",
 "https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d50_coco_384x288-01f3fbb9_20200727.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnetv1d101_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d101_coco_256x192-5bd08cab_20200727.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnetv1d101_coco_384x288.py",
 "https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d101_coco_384x288-5f9e421d_20200730.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnetv1d152_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d152_coco_256x192-c4df51dc_20200727.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/resnetv1d152_coco_384x288.py",
 "https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d152_coco_384x288-626c622d_20200730.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vipnas_mbv3_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/vipnas/vipnas_mbv3_coco_256x192-7018731a_20211122.pth"],
["/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vipnas_res50_coco_256x192.py",
 "https://download.openmmlab.com/mmpose/top_down/vipnas/vipnas_res50_coco_256x192-cc43b466_20210624.pth"],
]

TOPDOWN_POSEMODELS = [','.join(item) for item in TOPDOWN_POSEMODELS]
TOPDOWN_POSEMODELS = list(set(TOPDOWN_POSEMODELS))
TOPDOWN_POSEMODELS = [item.split(',') for item in TOPDOWN_POSEMODELS]