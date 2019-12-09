from PIL import Image

def grayscale(image):
    res = Image.new("RGB", (image.size[0], image.size[1]))
    new_data = []
    for pixel in image.getdata():
        r = pixel[0]
        g = pixel[1]
        b = pixel[2]
        value = (21 * r // 100) + (72 * g // 100) + (7 * b // 100) 
        new = (value, value, value)
        new_data.append(new)
    res.putdata(new_data)
    return res