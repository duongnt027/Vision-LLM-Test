from PIL import Image, ImageDraw

def string_to_list(query):
    lst = query.split(',')
    return [item.strip() for item in lst]

def image_from_url(url):
    return Image.open(requests.get(url, stream=True).raw)

def image_from_path(image_path):
    return Image.open(image_path)

def download_from_url(url):
    image_path = '/content/img.jpg'
    data = requests.get(url).content
    f = open(image_path,'wb')

    f.write(data)
    f.close()

    return image_path

def get_bboxes(output):
    out_str = output[0]

    # Replace single quotes with double quotes
    str_json = out_str.replace("'", '"')

    # Convert to JSON object
    bboxes = json.loads(str_json)

    return bboxes

def draw_box(image_path, bboxes):
    image = Image.open(image_path)

    for box in bboxes:
      draw = ImageDraw.Draw(image)
      draw.rectangle(box['bbox'], outline="red", width=2)
    return image