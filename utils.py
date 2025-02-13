from PIL import Image, ImageDraw
import requests

def string_to_list(query):
    lst = query.split(',')
    return [item.strip() for item in lst]

def image_from_url(url):
    return Image.open(requests.get(url, stream=True).raw)

def image_from_path(image_path):
    return Image.open(image_path)

def draw_bbox(image, boxes):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box, outline='red', width=2)
    return image

