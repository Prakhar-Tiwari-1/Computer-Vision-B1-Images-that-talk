from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import requests
import torch

def get_image_caption(image_path):
    """
    Generates a short caption for the provided image.

    Args:
        image_path (str): path to image file
    
    Returns:
        str: A string representing the caption for the image
    """

    image = Image.open(image_path).convert("RGB")
    model_name = "Salesforce/blip-image-captioning-large"
    device = "cpu" #cpu if you don't have gpu

    processor = BlipProcessor.from_pretrained(model_name, use_fast = True)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

    inputs = processor(image, return_tensors = 'pt').to(device)
    output = model.generate(**inputs, max_new_tokens = 20)

    caption = processor.decode(output[0], skip_special_tokens = True)

    return caption


def detect_objects(image_path):
    """
    Detects objects in the provided image 

    Args:
        image_path (str): path to image file
    
    Returns:
        str: A string representing bounding boxes of objects
    
    """
    image = Image.open(image_path).convert("RGB")

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    detections = ""

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detections += '[{}, {}, {}, {}]'.format(int(box[0]),int(box[1]),int(box[2]),int(box[3]))
        detections += " {}".format(model.config.id2label[int(label)])
        detections += " {}\n".format(float(score))
    
    return detections

if __name__ == '__main__':
    image_path = "D:\Downloads\Final Code Submission CV\calib.png"
    caption = get_image_caption(image_path)
    detections = detect_objects(image_path)
    print(caption)
    print(detections)
