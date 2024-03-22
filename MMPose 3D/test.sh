#!/bin/bash
rm /tmp/scenario/*
python body3d_pose_extractor.py --input mmpose_3d_examples/two_people_standing.jpg --output-root  /tmp/scenario --save-predictions
python body3d_pose_extractor.py --input /tmp/Scenario2-2_mp4.mp4 --output-root  /tmp/scenario --save-predictions
python body3d_pose_extractor.py --input /tmp/Scenario2-2_avi.avi --output-root  /tmp/scenario --save-predictions
python body3d_pose_extractor.py --input /tmp/basketball_short.mp4 --output-root  /tmp/scenario --save-predictions

