import time
from argparse import ArgumentParser
from functools import partial

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np

from mmpose.apis import (_track_by_iou, 
                         convert_keypoint_definition, extract_pose_sequence,
                         inference_pose_lifter_model, inference_topdown,
                         )
from mmpose.structures import (PoseDataSample, merge_data_samples)

from mmdet.apis import inference_detector

def process_one_image(detector, frame, frame_idx, pose_estimator,
                      pose_est_results_last, pose_est_results_list, next_id,
                      pose_lifter, visualize_frame, visualizer, show):
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
