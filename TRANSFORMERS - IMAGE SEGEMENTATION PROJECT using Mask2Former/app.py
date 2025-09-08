#
"""# Libraries"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.measure import regionprops

import torch
from PIL import Image
import requests

from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import gradio as gr


"""# Helper Function"""

def render_segmentation_overlay(prediction_output, input_image, segmentation_model, transparency=0.4):
    """
    Renders the segmentation mask on top of the original image,
    displaying colored regions for each segment and a legend indicating
    the object names and confidence scores.
    """
    # Extract segmentation mask array and metadata
    seg_mask = prediction_output["segmentation"].numpy()
    segment_metadata = prediction_output["segments_info"]
    img_height, img_width = seg_mask.shape

    # Initialize a blank RGB mask for coloring each segment
    rgb_overlay_mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    # Assign a random RGB color to each segment ID
    segment_colors = {}
    for metadata in segment_metadata:
        segment_colors[metadata["id"]] = np.random.randint(0, 255, size=3)

    # Apply colors to each segment in the overlay mask
    for metadata in segment_metadata:
        segment_pixels = seg_mask == metadata["id"]
        rgb_overlay_mask[segment_pixels] = segment_colors[metadata["id"]]

    # Convert original image to NumPy array
    img_array = np.asarray(input_image, dtype=np.uint8)

    # Blend original image with colored mask using transparency factor
    blended_image = (1 - transparency) * img_array + transparency * rgb_overlay_mask
    blended_image = blended_image.astype(np.uint8)

    # Display blended image
    plt.figure(figsize=(12, 8))
    plt.imshow(blended_image)
    axes = plt.gca()
    plt.axis("off")

    # Draw color legend on right side of the figure
    current_y = 10
    bar_height = 25
    vertical_spacing = 10
    for metadata in segment_metadata:
        label_text = segmentation_model.config.id2label[metadata["label_id"]]
        color_norm = segment_colors[metadata["id"]] / 255.0

        # Draw a colored rectangle
        rectangle = mpatches.Rectangle(
            (img_width - 30, current_y),
            20,
            bar_height,
            facecolor=color_norm,
            edgecolor=None,
            linewidth=0,
            transform=axes.transData,
            clip_on=False
        )
        axes.add_patch(rectangle)

        # Draw text label with object name and confidence score
        axes.text(
            img_width - 35,
            current_y + bar_height / 2,
            f"{label_text} ({metadata['score']:.2f})",
            va="center",
            ha="right",
            fontsize=11,
            color="white" if np.mean(color_norm) < 0.5 else "black",
            bbox=dict(facecolor=(0, 0, 0, 0.2), edgecolor="none", boxstyle="round,pad=0.2")
        )
        current_y += bar_height + vertical_spacing

    return blended_image


def plot_segmentation_labels(prediction_output, segmentation_model):
    """
    Visualizes a segmentation mask by plotting segment IDs and displaying
    the corresponding object labels at the centroid of each region.
    """
    # Retrieve the segmentation array
    seg_data = prediction_output["segmentation"]
    segment_metadata = prediction_output["segments_info"]

    # Convert to NumPy if needed
    seg_array = seg_data.numpy() if hasattr(seg_data, "numpy") else np.array(seg_data)


    # Iterate over segments to compute centroids and annotate labels
    for metadata in segment_metadata:
        seg_id = metadata["id"]
        label_id = metadata["label_id"]
        label_text = segmentation_model.config.id2label[label_id]

        # Create binary mask for current segment
        binary_mask = seg_array == seg_id

        # Use regionprops to find centroid
        props = regionprops(binary_mask.astype(np.uint8))
        if props:
            centroid_y, centroid_x = props[0].centroid
            axes.text(
                centroid_x,
                centroid_y,
                label_text,
                color="white",
                fontsize=8,
                weight="bold",
                ha="center",
                va="center",
                bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.2")
            )


def display_semantic_segmentation(semantic_map, base_image, segmentation_model, transparency=0.5):
    """
    Overlays a semantic segmentation mask on top of the base image.
    Each label is assigned a random color for visualization.
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    # Create a random color palette for all labels
    total_labels = len(segmentation_model.config.id2label)
    color_palette = np.random.randint(0, 255, (total_labels, 3))
    colored_mask = np.zeros((*semantic_map.shape, 3), dtype=np.uint8)

    # Assign colors to each unique label
    unique_labels = torch.unique(semantic_map)
    for label_value in unique_labels:
        colored_mask[semantic_map == label_value] = color_palette[label_value]

    # Convert original image and blend with colored mask
    base_array = np.array(base_image)
    blended_output = (1 - transparency) * base_array + transparency * colored_mask
    blended_output = blended_output.astype(np.uint8)

    return blended_output



"""# 1.0   SEMANTIC SEGMENTATION"""

def perform_semantic_segmentation(image_input_path):
    """
    Performs semantic segmentation inference using a pretrained Mask2Former model
    trained on the ADE20K dataset. It loads an image, processes it, and produces
    a segmentation label map resized to the original image dimensions.
    """
    # Specify the pretrained model checkpoint to use
    pretrained_checkpoint = "facebook/mask2former-swin-large-ade-semantic"

    # Load the AutoImageProcessor for preprocessing
    processor_instance = AutoImageProcessor.from_pretrained(pretrained_checkpoint)

    # Load the segmentation model
    segmentation_instance = Mask2FormerForUniversalSegmentation.from_pretrained(pretrained_checkpoint)

    # Load image from URL or local file

    loaded_image = image_input_path
    # Preprocess image to create input tensors for the model
    processed_inputs = processor_instance(images=loaded_image, return_tensors="pt")

    # Perform inference without tracking gradients
    with torch.no_grad():
        prediction_outputs = segmentation_instance(**processed_inputs)

    # Convert raw logits to semantic label map with original image size
    segmentation_map = processor_instance.post_process_semantic_segmentation(
        prediction_outputs,
        target_sizes=[loaded_image.size[::-1]]
    )[0]

    # Return the segmentation map, loaded image, and model instance
    return segmentation_map, loaded_image, segmentation_instance

'''
# Semantic segmentation
segmentation_map, loaded_image, segmentation_instance = perform_semantic_segmentation(url)

display_semantic_segmentation(segmentation_map, loaded_image, segmentation_instance, transparency=0.6)
display_semantic_segmentation(segmentation_map, loaded_image, segmentation_instance, transparency=1.0)
'''

"""# INSTANCE SEGMENTATION"""

def perform_instance_segmentation(image_input_path):
    """
    Performs instance segmentation on an input image using a pretrained Mask2Former model
    trained on the COCO dataset. It returns the segmentation results, the original image,
    and the model instance for downstream visualization.
    """
    # Define the pretrained checkpoint for COCO instance segmentation
    pretrained_checkpoint = "facebook/mask2former-swin-large-coco-instance"

    # Load the processor and the segmentation model
    processor_instance = AutoImageProcessor.from_pretrained(pretrained_checkpoint)
    segmentation_instance = Mask2FormerForUniversalSegmentation.from_pretrained(pretrained_checkpoint)

    loaded_image=image_input_path
    # Load the input image (supports URL or local file)
    #loaded_image = Image.fromarray(image_input_path).convert("RGB")

    # Preprocess image and create input tensors
    processed_inputs = processor_instance(images=loaded_image, return_tensors="pt")

    # Perform inference without gradients
    with torch.no_grad():
        prediction_outputs = segmentation_instance(**processed_inputs)

    # Convert raw model outputs into instance segmentation results
    instance_results = processor_instance.post_process_instance_segmentation(
        prediction_outputs,
        target_sizes=[loaded_image.size[::-1]]
    )[0]

    # Return the results, image, and model for visualization
    return instance_results, loaded_image, segmentation_instance

def perform_panoptic_segmentation(image_input_path):
    """
    Executes panoptic segmentation on an input image using a pretrained Mask2Former model
    trained on the COCO dataset. Returns segmentation results, the image object, and the model.
    """
    # Define the pretrained checkpoint for COCO panoptic segmentation
    pretrained_checkpoint = "facebook/mask2former-swin-base-coco-panoptic"

    # Load the processor and the segmentation model
    processor_instance = AutoImageProcessor.from_pretrained(pretrained_checkpoint)
    segmentation_instance = Mask2FormerForUniversalSegmentation.from_pretrained(pretrained_checkpoint)

    # Load the input image (supports URL or local file)
    if image_input_path.startswith("http"):
        loaded_image = Image.open(requests.get(image_input_path, stream=True).raw)
    else:
        loaded_image = Image.open(image_input_path)

    # Preprocess the image to create input tensors
    processed_inputs = processor_instance(images=loaded_image, return_tensors="pt")

    # Perform inference without computing gradients
    with torch.no_grad():
        prediction_outputs = segmentation_instance(**processed_inputs)

    # Convert raw model outputs into panoptic segmentation results
    panoptic_results = processor_instance.post_process_panoptic_segmentation(
        prediction_outputs,
        target_sizes=[loaded_image.size[::-1]]
    )[0]

    # Return results, image, and model
    return panoptic_results, loaded_image, segmentation_instance

def segment1(url):
    # Instance segmentation
    segmentation_map, loaded_image, segmentation_instance = perform_semantic_segmentation(url)
    #plot_segmentation_labels(instance_results, segmentation_instance)
    return display_semantic_segmentation(segmentation_map, loaded_image, segmentation_instance, transparency=0.6)

def segment2(url):
    # Instance segmentation
    instance_results, loaded_image, segmentation_instance = perform_instance_segmentation(url)

    #plot_segmentation_labels(instance_results, segmentation_instance)
    return render_segmentation_overlay(instance_results, loaded_image, segmentation_instance, transparency=0.5)

def segment3(url):
    # Instance segmentation
    instance_results, loaded_image, segmentation_instance = perform_instance_segmentation(url)

    #plot_segmentation_labels(instance_results, segmentation_instance)
    return render_segmentation_overlay(instance_results, loaded_image, segmentation_instance, transparency=0.5)


#####################################################
'''APP'''
####################################################

# Wrapper function to call appropriate segmentation
def segment_wrapper(image, mode):
    # Save input image temporarily (Gradio passes it as a NumPy array)
    image_pil = Image.fromarray(image).convert("RGB")

    if mode == "Semantic Segmentation":
        output = segment1(image_pil)
    elif mode == "Instance Segmentation":
        output = segment2(image_pil)
    elif mode == "Panoptic Segmentation":
        output = segment3(image_pil)
    else:
        raise ValueError("Unknown mode selected.")
    
    return output

# Gradio Interface
app = gr.Interface(
    fn=segment_wrapper,
    inputs=[
        gr.Image(type="numpy", label="Upload Image"),
        gr.Dropdown(
            choices=["Semantic Segmentation", "Instance Segmentation", "Panoptic Segmentation"],
            label="Segmentation Type",
            info="Choose the segmentation algorithm to apply"
        )
    ],
    outputs=gr.Image(label="Segmented Output"),
    title="IMAGE SEGEMENTATION PROJECT using Mask2FormerForUniversalSegmentation transformers",
    description="""
    ## Instructions
    1. Upload an image.
    2. Choose the segmentation type you want to apply.
    3. Click 'Submit' to process the image.
    4. The segmented result will appear below.
    """,
    theme=gr.themes.Soft(),  # Adds soft padding and modern styling
    css="""
    .gradio-container {padding: 2rem;}
    """
)

# Launch app
app.launch()