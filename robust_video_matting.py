import cv2
import numpy as np
import torch
import sys
sys.path.append('./RobustVideoMatting')

from RobustVideoMatting.inference import convert_video


model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3") # or "resnet50"


convert_video(
    model,                           # The loaded model, can be on any device (cpu or cuda).
    input_source="input.mp4",        # A video file or an image sequence directory.
    output_type='video',             # Choose "video" or "png_sequence"
    output_composition='com.mp4',    # File path if video; directory path if png sequence.
    output_alpha="pha.mp4",          # [Optional] Output the raw alpha prediction.
    output_foreground="fgr.mp4",     # [Optional] Output the raw foreground prediction.
    output_video_mbps=4,             # Output video mbps. Not needed for png sequence.
    seq_chunk=1,                    # Process n frames at once for better parallelism.
    device='cuda:0',
    progress=True                    # Print conversion progress.
)
