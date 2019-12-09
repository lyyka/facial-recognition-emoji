# OpenCV for image recognition
import cv2
import numpy as np

# Used to compile image path
import sys, os

# For image manipulation
from PIL import Image

# For arguments and CMD
import argparse

def draw_rect(image, faces):
    result_image = image.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return result_image

def image_over(image, over, faces):
    result_image = image.copy()
    
    for (x, y, w, h) in faces:
        this_detection_over = cv2.resize(over, (h, w))
        add_x, add_y = 0, 0
        for line in this_detection_over:
            add_x = 0
            add_y += 1
            for pixel in line:
                add_x += 1
                if pixel[3] == 0:
                    this_detection_over[add_y - 1, add_x - 1] = result_image[y + add_y - 1, x + add_x - 1]
        result_image[y:y+h, x:x+w] = this_detection_over

    return result_image

def blur_faces(image, faces):
    result_image = image.copy()

    # Loop through all faces found
    for (x, y, w, h) in faces:
        # Get the face pixels from starting image
        sub_face = image[y:y+h, x:x+w]

        # apply a gaussian blur on this new recangle image
        sub_face = cv2.GaussianBlur(sub_face, (23, 23), 30)

        # merge this blurry rectangle to our final image
        result_image[y:y+sub_face.shape[0], x:x+sub_face.shape[1]] = sub_face

    return result_image

def face_detect(image_path, args):
    # Read the cascade haar file
    faceCascade = cv2.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")

    # Read the image
    image  = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # Convert it to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Detect faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(10, 10),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    if(len(faces) > 0):
        # Smiley to be drawn over faces
        over = cv2.imread("./images/yeehaw.png", cv2.IMREAD_UNCHANGED)

        # Draw rect
        if args.mode == "rect":
            result = draw_rect(image, faces)

        # Blur faces
        if args.mode == "blur":
            result = blur_faces(image, faces)

        # Smiley over faces
        if args.mode == "emoji":
            result = image_over(image, over, faces)

        # Show the drawn image
        cv2.imshow("Faces found", result)
        cv2.waitKey(0)
    else:
        print("No faces found!")

if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()

    # add CLI arguments
    parser.add_argument('-m', '--mode', action="store", type=str,
                        help="blur, rect, emoji", required=True)
    parser.add_argument('-i', '--image', action="store", type=str, help="Image to be processed", required=True)

    # parse arguments
    args = parser.parse_args()

    # Get image name
    pathname = os.path.dirname(sys.argv[0])
    image_path = os.path.abspath(pathname) + "\\" + args.image

    face_detect(image_path, args)