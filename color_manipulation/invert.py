from PIL import Image

def invert(image):
    res = Image.new("RGB", (image.size[0], image.size[1]))
    new_data = []
    for pixel in image.getdata():
        r = 255 - pixel[0]
        g = 255 - pixel[1]
        b = 255 - pixel[2]
        new = (r, g, b)
        new_data.append(new)
    res.putdata(new_data)
    return res