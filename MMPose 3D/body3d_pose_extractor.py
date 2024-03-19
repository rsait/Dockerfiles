# Copyright (c) OpenMMLab. All rights reserved.
'''
python /tmp/Github/Ireland/body3d_pose_extractor.py  \
--input /tmp/Github/Ireland/two_people_standing.jpg \
--output-root  vis_results \
--save-predictions
'''
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
    parser.add_argument('--input', type=str, default='/tmp/Github/Ireland/two_people_standing.jpg', help='Video path')
    parser.add_argument(
        '--output-root',
        type=str,
        default='vis_results',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=True,
        help='Whether to save predicted results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    return args


def process_one_image(detector, frame, frame_idx, pose_estimator,
                      pose_est_results_last, pose_est_results_list, next_id,
                      pose_lifter, visualize_frame, visualizer):
    """Visualize detected and predicted keypoints of one image.

    Pipeline:
    1. Detect keypoints with `detector`.
    2. Estimate 2D poses with `pose_estimator`.
    3. Convert 2D keypoints for pose lifting.
    4. Extract pose sequence with `extract_pose_sequence`.
    5. Lift poses to 3D with `pose_lifter`.
    6. Apply post-processing.
    7. Visualize results with `visualizer`.

    Args:
        args (Argument): Custom command-line arguments.
        detector (mmdet.BaseDetector): The mmdet detector.
        frame (np.ndarray): The image frame read from input image or video.
        frame_idx (int): The index of current frame.
        pose_estimator (TopdownPoseEstimator): The pose estimator for 2d pose.
        pose_est_results_last (list(PoseDataSample)): The results of pose
            estimation from the last frame for tracking instances.
        pose_est_results_list (list(list(PoseDataSample))): The list of all
            pose estimation results converted by
            ``convert_keypoint_definition`` from previous frames. In
            pose-lifting stage it is used to obtain the 2d estimation sequence.
        next_id (int): The next track id to be used.
        pose_lifter (PoseLifter): The pose-lifter for estimating 3d pose.
        visualize_frame (np.ndarray): The image for drawing the results on.
        visualizer (Visualizer): The visualizer for visualizing the 2d and 3d
            pose estimation results.

    Returns:
        pose_est_results (list(PoseDataSample)): The pose estimation result of
            the current frame.
        pose_est_results_list (list(list(PoseDataSample))): The list of all
            converted pose estimation results until the current frame.
        pred_3d_instances (InstanceData): The result of pose-lifting.
            Specifically, the predicted keypoints and scores are saved at
            ``pred_3d_instances.keypoints`` and
            ``pred_3d_instances.keypoint_scores``.
        next_id (int): The next track id to be used.
    """
    pose_lift_dataset = pose_lifter.cfg.test_dataloader.dataset
    pose_lift_dataset_name = pose_lifter.dataset_meta['dataset_name']

    # First stage: conduct 2D pose detection in a Topdown manner
    # use detector to obtain person bounding boxes
    det_result = inference_detector(detector, frame)
    pred_instance = det_result.pred_instances.cpu().numpy()

    # filter out the person instances with category and bbox threshold
    # e.g. 0 for person in COCO
    bboxes = pred_instance.bboxes
    bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                   pred_instance.scores > 0.3)]

    # estimate pose results for current image
    pose_est_results = inference_topdown(pose_estimator, frame, bboxes)

    _track = _track_by_iou

    pose_det_dataset_name = pose_estimator.dataset_meta['dataset_name']
    pose_est_results_converted = []

    # convert 2d pose estimation results into the format for pose-lifting
    # such as changing the keypoint order, flipping the keypoint, etc.
    for i, data_sample in enumerate(pose_est_results):
        pred_instances = data_sample.pred_instances.cpu().numpy()
        keypoints = pred_instances.keypoints
        # calculate area and bbox
        if 'bboxes' in pred_instances:
            areas = np.array([(bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                              for bbox in pred_instances.bboxes])
            pose_est_results[i].pred_instances.set_field(areas, 'areas')
        else:
            areas, bboxes = [], []
            for keypoint in keypoints:
                xmin = np.min(keypoint[:, 0][keypoint[:, 0] > 0], initial=1e10)
                xmax = np.max(keypoint[:, 0])
                ymin = np.min(keypoint[:, 1][keypoint[:, 1] > 0], initial=1e10)
                ymax = np.max(keypoint[:, 1])
                areas.append((xmax - xmin) * (ymax - ymin))
                bboxes.append([xmin, ymin, xmax, ymax])
            pose_est_results[i].pred_instances.areas = np.array(areas)
            pose_est_results[i].pred_instances.bboxes = np.array(bboxes)

        # track id
        track_id, pose_est_results_last, _ = _track(data_sample,
                                                    pose_est_results_last,
                                                    0.3)
        if track_id == -1:
            if np.count_nonzero(keypoints[:, :, 1]) >= 3:
                track_id = next_id
                next_id += 1
            else:
                # If the number of keypoints detected is small,
                # delete that person instance.
                keypoints[:, :, 1] = -10
                pose_est_results[i].pred_instances.set_field(
                    keypoints, 'keypoints')
                pose_est_results[i].pred_instances.set_field(
                    pred_instances.bboxes * 0, 'bboxes')
                pose_est_results[i].set_field(pred_instances, 'pred_instances')
                track_id = -1
        pose_est_results[i].set_field(track_id, 'track_id')

        # convert keypoints for pose-lifting
        pose_est_result_converted = PoseDataSample()
        pose_est_result_converted.set_field(
            pose_est_results[i].pred_instances.clone(), 'pred_instances')
        pose_est_result_converted.set_field(
            pose_est_results[i].gt_instances.clone(), 'gt_instances')
        keypoints = convert_keypoint_definition(keypoints,
                                                pose_det_dataset_name,
                                                pose_lift_dataset_name)
        pose_est_result_converted.pred_instances.set_field(
            keypoints, 'keypoints')
        pose_est_result_converted.set_field(pose_est_results[i].track_id,
                                            'track_id')
        pose_est_results_converted.append(pose_est_result_converted)

    pose_est_results_list.append(pose_est_results_converted.copy())

    # Second stage: Pose lifting
    # extract and pad input pose2d sequence
    pose_seq_2d = extract_pose_sequence(
        pose_est_results_list,
        frame_idx=frame_idx,
        causal=pose_lift_dataset.get('causal', False),
        seq_len=pose_lift_dataset.get('seq_len', 1),
        step=pose_lift_dataset.get('seq_step', 1))

    # conduct 2D-to-3D pose lifting
    pose_lift_results = inference_pose_lifter_model(
        pose_lifter,
        pose_seq_2d,
        image_size=visualize_frame.shape[:2],
        norm_pose_2d=True)

    # post-processing
    for idx, pose_lift_result in enumerate(pose_lift_results):
        pose_lift_result.track_id = pose_est_results[idx].get('track_id', 1e4)

        pred_instances = pose_lift_result.pred_instances
        keypoints = pred_instances.keypoints
        keypoint_scores = pred_instances.keypoint_scores
        if keypoint_scores.ndim == 3:
            keypoint_scores = np.squeeze(keypoint_scores, axis=1)
            pose_lift_results[
                idx].pred_instances.keypoint_scores = keypoint_scores
        if keypoints.ndim == 4:
            keypoints = np.squeeze(keypoints, axis=1)

        keypoints = keypoints[..., [0, 2, 1]]
        keypoints[..., 0] = -keypoints[..., 0]
        keypoints[..., 2] = -keypoints[..., 2]

        # rebase height (z-axis)
        keypoints[..., 2] -= np.min(
            keypoints[..., 2], axis=-1, keepdims=True)

        pose_lift_results[idx].pred_instances.keypoints = keypoints

    pose_lift_results = sorted(
        pose_lift_results, key=lambda x: x.get('track_id', 1e4))

    pred_3d_data_samples = merge_data_samples(pose_lift_results)
    det_data_sample = merge_data_samples(pose_est_results)
    pred_3d_instances = pred_3d_data_samples.get('pred_instances', None)

    # Prepare the 2D bounding boxes for return
    bboxes_2d = np.array([data_sample.pred_instances.bboxes for data_sample in pose_est_results])

    # Visualization
    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            visualize_frame,
            data_sample=pred_3d_data_samples,
            det_data_sample=det_data_sample,
            draw_gt=False,
            dataset_2d=pose_det_dataset_name,
            dataset_3d=pose_lift_dataset_name,
            show=show,
            draw_bbox=True,
            kpt_thr=0.3,
            num_instances=1,
            wait_time=0)

    return pose_est_results, pose_est_results_list, pred_3d_instances, next_id, bboxes_2d


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

        if save_output:
            frame_vis = visualizer.get_image()
            mmcv.imwrite(mmcv.rgb2bgr(frame_vis), output_file)

    elif input_type in ['video']:
        next_id = 0
        pose_est_results = []

        video = cv2.VideoCapture(args.input)

        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        if int(major_ver) < 3:
            fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        else:
            fps = video.get(cv2.CAP_PROP_FPS)

        video_writer = None
        frame_idx = 0

        while video.isOpened():
            success, frame = video.read()
            frame_idx += 1

            if not success:
                break

            pose_est_results_last = pose_est_results

            # First stage: 2D pose detection
            # make person results for current image
            (pose_est_results, pose_est_results_list, pred_3d_instances,
             next_id, bboxes_2d) = process_one_image(
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
                        instances=split_instances(pred_3d_instances)))
                bboxes_2d_list.append(
                    dict(
                        frame_id=frame_idx,
                        bboxes_2d=bboxes_2d))

            if save_output:
                frame_vis = visualizer.get_image()
                if video_writer is None:
                    # the size of the image with visualization may vary
                    # depending on the presence of heatmaps
                    video_writer = cv2.VideoWriter(output_file, fourcc, fps,
                                                   (frame_vis.shape[1],
                                                    frame_vis.shape[0]))

                video_writer.write(mmcv.rgb2bgr(frame_vis))

            if show:
                # press ESC to exit
                if cv2.waitKey(5) & 0xFF == 27:
                    break
                time.sleep(0)

        video.release()

        if video_writer:
            video_writer.release()
    else:
        args.save_predictions = False
        raise ValueError(
            f'file {os.path.basename(args.input)} has invalid format.')

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
        input_type = input_type.replace('webcam', 'video')
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
with open(pred_save_pickle_path, 'rb') as f:
    pred_instances_list, bboxes_2d_list = pkl.load(f)
print(pred_instances_list[0])
'''