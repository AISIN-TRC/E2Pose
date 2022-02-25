## coding: UTF-8
import numpy as np


# Ref. https://github.com/openpifpaf/openpifpaf/blob/2837cadde2898428d9d36fa5af41c708b1af81ab/src/openpifpaf/plugins/posetrack/constants.py
POSE_DATASETS ={
    'COCO': {
        'dir_train':'/dataset/COCO/train2017',
        'dir_val':'/dataset/COCO/val2017',
        'dir_test':'/dataset/COCO/test2017',
        'json_train': '/dataset/COCO/annotations/person_keypoints_train2017.json',
        'json_val': '/dataset/COCO/annotations/person_keypoints_val2017.json',
        'json_testdev': '/dataset/COCO/annotations/person_keypoints_test-dev-2017.json',
        'with_mask': True,
        'joints': {0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear', 5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow',8: 'right_elbow',
                    9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip', 13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle'},
        'joints_flip': {0: 'nose', 1: 'right_eye', 2: 'left_eye', 3: 'right_ear', 4: 'left_ear', 5: 'right_shoulder', 6: 'left_shoulder', 7: 'right_elbow',8: 'left_elbow',
                    9: 'right_wrist', 10: 'left_wrist', 11: 'right_hip', 12: 'left_hip', 13: 'right_knee', 14: 'left_knee', 15: 'right_ankle', 16: 'left_ankle'},
        'th_size':None,
        'skeleton':np.array([[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]])-1,
        'oks_sigma': np.array([0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089]),
        'upright_pose': np.array([
            [0.0, 9.3, 2.0],  # 'nose',            # 1
            [-0.35, 9.7, 2.0],  # 'left_eye',        # 2
            [0.35, 9.7, 2.0],  # 'right_eye',       # 3
            [-0.7, 9.5, 2.0],  # 'left_ear',        # 4
            [0.7, 9.5, 2.0],  # 'right_ear',       # 5
            [-1.4, 8.0, 2.0],  # 'left_shoulder',   # 6
            [1.4, 8.0, 2.0],  # 'right_shoulder',  # 7
            [-1.75, 6.0, 2.0],  # 'left_elbow',      # 8
            [1.75, 6.2, 2.0],  # 'right_elbow',     # 9
            [-1.75, 4.0, 2.0],  # 'left_wrist',      # 10
            [1.75, 4.2, 2.0],  # 'right_wrist',     # 11
            [-1.26, 4.0, 2.0],  # 'left_hip',        # 12
            [1.26, 4.0, 2.0],  # 'right_hip',       # 13
            [-1.4, 2.0, 2.0],  # 'left_knee',       # 14
            [1.4, 2.1, 2.0],  # 'right_knee',      # 15
            [-1.4, 0.0, 2.0],  # 'left_ankle',      # 16
            [1.4, 0.1, 2.0],  # 'right_ankle',     # 17
        ]),
    },
}
