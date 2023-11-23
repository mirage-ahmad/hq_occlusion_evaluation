import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import VitMatteForImageMatting, VitMatteImageProcessor
from huggingface_hub import hf_hub_download
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import matplotlib.pyplot as plt
import random

# Load the VitMatte model and processor
vitmat_processor = VitMatteImageProcessor.from_pretrained("hustvl/vitmatte-base-composition-1k")
vitmat_model = VitMatteForImageMatting.from_pretrained("hustvl/vitmatte-base-composition-1k")

segformer_processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
segformer_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0")


def visualize_segmentation(image, mask, alpha=0.5):
    """
    Visualize the segmentation mask on the original image.

    :param image: PIL Image, the original image.
    :param mask: numpy.ndarray, the segmentation mask.
    :param alpha: float, the transparency level of the overlay.
    """
    # Assign random colors to each class
    num_classes = np.unique(mask).size
    colors = np.array([random.sample(range(256), 3) for _ in range(num_classes)])

    # Colorize the mask
    mask_color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls in range(num_classes):
        mask_color[mask == cls] = colors[cls]

    # Convert mask to PIL for easy overlay
    mask_image = Image.fromarray(mask_color).resize(image.size)

    # Overlay the mask on the original image
    overlayed_image = Image.blend(image, mask_image, alpha=alpha)

    cv2.imshow('segmentation', np.array(overlayed_image))

    # # Display the images
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.imshow(image)
    # plt.title("Original Image")
    # plt.axis('off')
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(overlayed_image)
    # plt.title("Segmentation Overlay")
    # plt.axis('off')
    #
    # plt.show()


def create_trimap(mask, dilate_size=30, erode_size=5):
    # Dilate and erode the mask to create the unknown region
    kernel = np.ones((dilate_size, dilate_size), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)

    cv2.imshow('dilated', 255 * dilated)

    kernel = np.ones((erode_size, erode_size), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    cv2.imshow('eroded', 255 * eroded)

    trimap = np.where(dilated != eroded, 128, mask * 255).astype(np.uint8)
    cv2.imshow('trimap', trimap)

    return trimap


def process_frame(frame):
    # Convert frame to PIL Image
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Process the frame through Segformer
    inputs = segformer_processor(images=frame_pil, return_tensors="pt")
    outputs = segformer_model(**inputs)
    logits = outputs.logits

    # First, rescale logits to original image size
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=frame_pil.size[::-1],  # (height, width)
        mode='bilinear',
        align_corners=False
    )

    # Second, apply argmax on the class dimension
    pred_seg = upsampled_logits.argmax(dim=1)[0]

    visualize_segmentation(frame_pil, pred_seg)

    # Convert logits to binary mask (assuming a certain class as foreground)
    binary_mask = pred_seg != 0  # Replace with the ID of your desired foreground class
    binary_mask = binary_mask.cpu().numpy().astype(np.uint8)

    # Generate trimap from binary mask
    trimap = create_trimap(binary_mask)

    cv2.imshow('frame', frame)
    cv2.imshow('trimap', trimap)
    cv2.imshow('binary', 255 * binary_mask)
    cv2.waitKey(1)

    # Process with VitMatte
    trimap_pil = Image.fromarray(trimap).convert("L")
    inputs = vitmat_processor(images=frame_pil, trimaps=trimap_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = vitmat_model(**inputs)
        alphas = outputs.alphas

    alpha_image = nn.functional.interpolate(
        alphas,
        size=frame_pil.size[::-1],  # (height, width)
        mode='bilinear',
        align_corners=False
    )

    alpha_image = alpha_image.cpu().numpy().squeeze() / 255

    alpha_image = cv2.merge([alpha_image, alpha_image, alpha_image])

    cv2.imshow('matted', alpha_image * frame)

    # Convert the result back to an image
    processed_image = Image.fromarray(alphas.squeeze().cpu().numpy())

    # Convert back to NumPy array and correct color format
    processed_frame = np.array(processed_image)
    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

    return processed_frame


# Open the video file
video_path = 's_e9d7f0e1-3a8f-38ef-ba86-8a7eca835af5_v_Linguana-PoC-AzhytlKRga4-MX.mp4'
cap = cv2.VideoCapture(video_path)

# Define the codec and create VideoWriter object to save the output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame
    processed_frame = process_frame(frame)

    # Display the processed frame
    cv2.imshow('Processed Frame', np.array(processed_frame).astype(np.uint8))

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Write the frame into the output file (additional processing may be required here)
    out.write(processed_frame.astype(np.uint8))

# Release everything when the job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
