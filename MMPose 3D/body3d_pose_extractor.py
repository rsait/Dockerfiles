# Copyright (c) OpenMMLab. All rights reserved.
import logging
import mimetypes
import os
import time
from argparse import ArgumentParser
from functools import partial
import pandas as pd
import pickle as pkl

from processimage import process_one_image

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log

from mmpose.apis import (_track_by_iou, _track_by_oks,
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

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--det_config', 
                        default='demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py', 
                        help='Config file for detection')
    parser.add_argument('--det_checkpoint', 
                        default='https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth', 
                        help='Checkpoint file for detection')
    parser.add_argument(
        '--pose_estimator_config',
        type=str,
        default='configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py',
        help='Config file for the 1st stage 2D pose estimator')
    parser.add_argument(
        '--pose_estimator_checkpoint',
        type=str,
        default='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth',
        help='Checkpoint file for the 1st stage 2D pose estimator')
    parser.add_argument(
        '--pose_lifter_config',
        default='configs/body_3d_keypoint/video_pose_lift/h36m/video-pose-lift_tcn-243frm-supv-cpn-ft_8xb128-200e_h36m.py',
        help='Config file for the 2nd stage pose lifter model')
    parser.add_argument(
        '--pose_lifter_checkpoint',
        default='https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth',
        help='Checkpoint file for the 2nd stage pose lifter model')
    parser.add_argument('--input', type=str, default='', help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='Whether to show visualizations')
    parser.add_argument(
        '--disable-rebase-keypoint',
        action='store_true',
        default=False,
        help='Whether to disable rebasing the predicted 3D pose so its '
        'lowest keypoint has a height of 0 (landing on the ground). Rebase '
        'is useful for visualization when the model do not predict the '
        'global position of the 3D pose.')
    parser.add_argument(
        '--disable-norm-pose-2d',
        action='store_true',
        default=False,
        help='Whether to scale the bbox (along with the 2D pose) to the '
        'average bbox scale of the dataset, and move the bbox (along with the '
        '2D pose) to the average bbox center of the dataset. This is useful '
        'when bbox is small, especially in multi-person scenarios.')
    parser.add_argument(
        '--num-instances',
        type=int,
        default=1,
        help='The number of 3D poses to be visualized in every frame. If '
        'less than 0, it will be set to the number of pose results in the '
        'first frame.')
    parser.add_argument(
        '--output-root',
        type=str,
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=False,
        help='Whether to save predicted results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument('--kpt-thr', type=float, default=0.3)
    parser.add_argument(
        '--use-oks-tracking', action='store_true', help='Using OKS tracking')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--online',
        action='store_true',
        default=False,
        help='Inference mode. If set to True, can not use future frame'
        'information when using multi frames for inference in the 2D pose'
        'detection stage. Default: False.')

    args = parser.parse_args()
    return args

def main():
    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parse_args()

    assert args.show or (args.output_root != '')
    assert args.input != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    pose_estimator = init_model(
        args.pose_estimator_config,
        args.pose_estimator_checkpoint,
        device=args.device.lower())

    assert isinstance(pose_estimator, TopdownPoseEstimator), 'Only "TopDown"' \
        'model is supported for the 1st stage (2D pose detection)'

    det_kpt_color = pose_estimator.dataset_meta.get('keypoint_colors', None)
    det_dataset_skeleton = pose_estimator.dataset_meta.get(
        'skeleton_links', None)
    det_dataset_link_color = pose_estimator.dataset_meta.get(
        'skeleton_link_colors', None)

    pose_lifter = init_model(
        args.pose_lifter_config,
        args.pose_lifter_checkpoint,
        device=args.device.lower())

    assert isinstance(pose_lifter, PoseLifter), \
        'Only "PoseLifter" model is supported for the 2nd stage ' \
        '(2D-to-3D lifting)'

    pose_lifter.cfg.visualizer.radius = args.radius
    pose_lifter.cfg.visualizer.line_width = args.thickness
    pose_lifter.cfg.visualizer.det_kpt_color = det_kpt_color
    pose_lifter.cfg.visualizer.det_dataset_skeleton = det_dataset_skeleton
    pose_lifter.cfg.visualizer.det_dataset_link_color = det_dataset_link_color
    visualizer = VISUALIZERS.build(pose_lifter.cfg.visualizer)

    # the dataset_meta is loaded from the checkpoint
    visualizer.set_dataset_meta(pose_lifter.dataset_meta)

    if args.input == 'webcam':
        input_type = 'webcam'
    else:
        input_type = mimetypes.guess_type(args.input)[0].split('/')[0]

    if args.output_root == '':
        save_output = False
    else:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root,
                                   os.path.basename(args.input))
        if args.input == 'webcam':
            output_file += '.mp4'
        save_output = True

    if args.save_predictions:
        assert args.output_root != ''
        args.pred_save_json_path = f'{args.output_root}/results_' \
            f'{os.path.splitext(os.path.basename(args.input))[0]}.json'
        args.pred_save_dicts_path = f'{args.output_root}/results_' \
            f'{os.path.splitext(os.path.basename(args.input))[0]}_dicts.pkl'
        args.pred_save_df_path = f'{args.output_root}/results_' \
            f'{os.path.splitext(os.path.basename(args.input))[0]}_df.pkl'

    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    pose_est_results_list = []
    pred_instances_list = []
    if input_type == 'image':
        frame = mmcv.imread(args.input, channel_order='rgb')
        pose_est_results, pose_est_results_list, pred_3d_instances, _, bboxes_2d = process_one_image(
            args=args,
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
            instances_3d = split_instances(pred_3d_instances)
            instances_2d=[{
                            'keypoints': data_sample.pred_instances.keypoints,
                            'keypoint_scores': data_sample.pred_instances.keypoint_scores,
                            'track_id': data_sample.get('track_id', None)
                        } for data_sample in pose_est_results]
            pred_instances_list.append(dict(
                frame_id=0,
                instances_3d=instances_3d,
                instances_2d=instances_2d,
                bboxes_2d=bboxes_2d
            ))

        if save_output:
            frame_vis = visualizer.get_image()
            mmcv.imwrite(mmcv.rgb2bgr(frame_vis), output_file)

    elif input_type in ['webcam', 'video']:
        next_id = 0
        pose_est_results = []
        frame_count = 0

        if args.input == 'webcam':
            video = cv2.VideoCapture(0)
        else:
            video = cv2.VideoCapture(args.input)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            print_log(
                f'Video {args.input} has {frame_count} frames in total.',
                logger='current',
                level=logging.INFO)

        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        if int(major_ver) < 3:
            fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        else:
            fps = video.get(cv2.CAP_PROP_FPS)

        video_writer = None
        frame_idx = 0

        # Variables to track whether resizing is needed and the target dimensions
        need_to_resize = False
        target_width, target_height = None, None

        while video.isOpened():
            success, frame = video.read()
            frame_idx += 1
            print_log(f'Processing frame {frame_idx} / {frame_count}',
                logger='current',
                level=logging.INFO) 

            if not success:
                break

            pose_est_results_last = pose_est_results

            # First stage: 2D pose detection
            # make person results for current image
            (pose_est_results, pose_est_results_list, pred_3d_instances,
             next_id, bboxes_2d) = process_one_image(
                 args=args,
                 detector=detector,
                 frame=frame,
                 frame_idx=frame_idx,
                 pose_estimator=pose_estimator,
                 pose_est_results_last=pose_est_results_last,
                 pose_est_results_list=pose_est_results_list,
                 next_id=next_id,
                 pose_lifter=pose_lifter,
                 visualize_frame=mmcv.bgr2rgb(frame),
                 visualizer=visualizer)

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
                        } for data_sample in pose_est_results],  # Add 2D keypoints and scores here
                        bboxes_2d=bboxes_2d
                    )
                )
            if save_output:
                frame_vis = visualizer.get_image()
                if video_writer is None:
                    # Check dimensions only for the first frame to decide the video writer's dimensions
                    max_width = 1920
                    max_height = 1080
                    original_width, original_height = frame_vis.shape[1], frame_vis.shape[0]

                    # Determine if resizing is necessary and the target dimensions
                    if original_width > max_width or original_height > max_height:
                        need_to_resize = True
                        target_width = original_width // 2
                        target_height = original_height // 2
                    else:
                        target_width = original_width
                        target_height = original_height

                    # Initialize the video writer with the determined dimensions
                    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (target_width, target_height))

                # Resize the frame if necessary (for all frames if the first frame required resizing)
                if need_to_resize:
                    frame_vis = cv2.resize(frame_vis, (target_width, target_height))

                video_writer.write(mmcv.rgb2bgr(frame_vis))

            if args.show:
                # press ESC to exit
                if cv2.waitKey(5) & 0xFF == 27:
                    break
                time.sleep(args.show_interval)

        video.release()

        if video_writer:
            video_writer.release()
    else:
        args.save_predictions = False
        raise ValueError(
            f'file {os.path.basename(args.input)} has invalid format.')

    if args.save_predictions:
        with open(args.pred_save_json_path, 'w') as f:
            json.dump(
                dict(
                    meta_info=pose_lifter.dataset_meta,
                    instance_info=pred_instances_list),
                f,
                indent='\t')
        print(f'predictions have been saved at {args.pred_save_json_path}')

        # This list will hold one dictionary for each skeleton across all frames
        skeletons_data = []

        for data in pred_instances_list:
            frame_id = data['frame_id']
            bboxes_2d = data['bboxes_2d']

            for instance_3d, instance_2d, bbox_2d in zip(data['instances_3d'], data['instances_2d'], bboxes_2d):
                skeleton_data = {
                    'frame_id': frame_id,
                    '3d_keypoints': instance_3d['keypoints'],
                    '3d_keypoints_scores': instance_3d['keypoint_scores'],
                    '2d_keypoints': instance_2d['keypoints'][0].tolist(),  # Convert numpy array to list for DataFrame compatibility
                    '2d_keypoints_scores': instance_2d['keypoint_scores'][0].tolist(),  # Convert numpy array to list
                    'track_id': instance_2d['track_id'],
                    'bboxes_2d': bbox_2d[0].tolist()  # Convert numpy array to list
                }
                skeletons_data.append(skeleton_data)

        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(skeletons_data)
        with open(args.pred_save_dicts_path, 'wb') as f:
            pkl.dump(pred_instances_list, f)
        print(f'predictions have been saved at {args.pred_save_dicts_path}')

        with open(args.pred_save_df_path, 'wb') as f:
            pkl.dump(df, f)
        print(f'predictions have been saved at {args.pred_save_df_path}')

    if save_output:
        input_type = input_type.replace('webcam', 'video')
        print_log(
            f'the output {input_type} has been saved at {output_file}',
            logger='current',
            level=logging.INFO)


if __name__ == '__main__':
    main()