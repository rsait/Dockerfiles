# Copyright (c) OpenMMLab. All rights reserved.
'''
python body3d_pose_extractor.py  \
--input mmpose_3d_examples/two_people_standing.jpg \
--output-root  /tmp \
--save-predictions
'''

from processimage import process_one_image

import logging
import mimetypes
import os
import time
from argparse import ArgumentParser
from functools import partial
import pickle as pkl

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log

from mmpose.apis import (_track_by_iou,
                         convert_keypoint_definition, extract_pose_sequence,
                         inference_pose_lifter_model, inference_topdown,
                         init_model)
from mmpose.models.pose_estimators import PoseLifter
from mmpose.models.pose_estimators.topdown import TopdownPoseEstimator
from mmpose.registry import VISUALIZERS
from mmpose.structures import (PoseDataSample, merge_data_samples,
                               split_instances)
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

det_config='demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py'
det_checkpoint='https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'
pose_estimator_config='configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py'
pose_estimator_checkpoint='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth'
pose_lifter_config='configs/body_3d_keypoint/video_pose_lift/h36m/video-pose-lift_tcn-243frm-supv-cpn-ft_8xb128-200e_h36m.py'
pose_lifter_checkpoint='https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth'
show = False
online = False
device ='cuda:0'

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, default='mmpose_3d_examples/two_people_standing.jpg', help='Image path')
    parser.add_argument(
        '--output-root',
        type=str,
        default='vis_results',
        help='Root of the output image file. '
        'Default not saving the visualization image.')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=True,
        help='Whether to save predicted results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    return args

def main():
    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parse_args()

    assert show or (args.output_root != '')
    assert args.input != ''

    print(args)

    detector = init_detector(
        det_config, det_checkpoint, device=device.lower())
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    pose_estimator = init_model(
        pose_estimator_config,
        pose_estimator_checkpoint,
        device=device.lower())

    assert isinstance(pose_estimator, TopdownPoseEstimator), 'Only "TopDown"' \
        'model is supported for the 1st stage (2D pose detection)'

    det_kpt_color = pose_estimator.dataset_meta.get('keypoint_colors', None)
    det_dataset_skeleton = pose_estimator.dataset_meta.get(
        'skeleton_links', None)
    det_dataset_link_color = pose_estimator.dataset_meta.get(
        'skeleton_link_colors', None)

    pose_lifter = init_model(
        pose_lifter_config,
        pose_lifter_checkpoint,
        device=device.lower())

    assert isinstance(pose_lifter, PoseLifter), \
        'Only "PoseLifter" model is supported for the 2nd stage ' \
        '(2D-to-3D lifting)'

    pose_lifter.cfg.visualizer.radius = 3
    pose_lifter.cfg.visualizer.line_width = 1
    pose_lifter.cfg.visualizer.det_kpt_color = det_kpt_color
    pose_lifter.cfg.visualizer.det_dataset_skeleton = det_dataset_skeleton
    pose_lifter.cfg.visualizer.det_dataset_link_color = det_dataset_link_color
    visualizer = VISUALIZERS.build(pose_lifter.cfg.visualizer)

    # the dataset_meta is loaded from the checkpoint
    visualizer.set_dataset_meta(pose_lifter.dataset_meta)

    input_type = mimetypes.guess_type(args.input)[0].split('/')[0]

    if args.output_root == '':
        save_output = False
    else:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root,
                                   os.path.basename(args.input))
        save_output = True

    if args.save_predictions:
        assert args.output_root != ''
        pred_save_path = f'{args.output_root}/results_' \
            f'{os.path.splitext(os.path.basename(args.input))[0]}.json'
        pred_save_pickle_path = f'{args.output_root}/results_' \
            f'{os.path.splitext(os.path.basename(args.input))[0]}.pkl'

    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    pose_est_results_list = []
    pred_instances_list = []
    bboxes_2d_list = []
    if input_type == 'image':
        frame = mmcv.imread(args.input, channel_order='rgb')
        _, _, pred_3d_instances, _ , bboxes_2d = process_one_image(
            detector=detector,
            frame=frame,
            frame_idx=0,
            pose_estimator=pose_estimator,
            pose_est_results_last=[],
            pose_est_results_list=pose_est_results_list,
            next_id=0,
            pose_lifter=pose_lifter,
            visualize_frame=frame,
            visualizer=visualizer)

        if args.save_predictions:
            # save prediction results
            pred_instances_list = split_instances(pred_3d_instances)
            bboxes_2d_list = bboxes_2d




            if args.save_predictions:
                # save prediction results
                pred_instances_list.append(
                    dict(
                        frame_id=frame_idx,
                        instances_3d=split_instances(pred_3d_instances),
                        instances_2d=[{
                            'keypoints': data_sample.pred_instances.keypoints,
                            'keypoint_scores': data_sample.pred_instances.keypoint_scores,
                            'track_id': data_sample.get('track_id', None)
                        } for data_sample in pose_est_results]  # Add 2D keypoints and scores here
                    )
                )
                bboxes_2d_list.append(
                    dict(
                        frame_id=frame_idx,
                        bboxes_2d=bboxes_2d))




        if save_output:
            frame_vis = visualizer.get_image()
            mmcv.imwrite(mmcv.rgb2bgr(frame_vis), output_file)

    if args.save_predictions:
        with open(pred_save_path, 'w') as f:
            json.dump(
                dict(
                    meta_info=pose_lifter.dataset_meta,
                    instance_info=(pred_instances_list, bboxes_2d_list)),
                f,
                indent='\t')
        print(f'predictions have been saved at {pred_save_path}')
        with open(pred_save_pickle_path, 'wb') as f:
            pkl.dump((pred_instances_list, bboxes_2d_list),f)

    if save_output:
        print_log(
            f'the output {input_type} has been saved at {output_file}',
            logger='current',
            level=logging.INFO)


if __name__ == '__main__':
    main()

# load pickle file
'''
import pickle as pkl
pred_save_pickle_path = "vis_results/results_two_people_standing.pkl"
pred_save_pickle_path = "vis_results/results_164970135-b14e424c-765a-4180-9bc8-fa8d6abc5510.pkl"
pred_save_pickle_path = "~/mmpose_3d_tmp/results_two_people_standing.pkl"
pred_save_pickle_path = "results_two_people_standing.pkl"
with open(pred_save_pickle_path, 'rb') as f:
    pred_instances_list, bboxes_2d_list = pkl.load(f)
print(pred_instances_list[0])
'''