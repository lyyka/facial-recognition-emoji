# OpenCV for image recognition
import cv2
import numpy as np

# Used to compile image path
import sys, os, time

# For image manipulation
from PIL import Image

# For arguments and CMD
import argparse

def draw_rect(image, faces):
    result_image = image.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return result_image

def image_over(smileFaceCascade, emojis, image, faces):
    result_image = image.copy()
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2BGRA)
    gray = cv2.cvtColor(result_image, cv2.COLOR_BGRA2GRAY)
    
    for (x, y, w, h) in faces:
        # See if there are smileys to switch emojis
        smiles = smileFaceCascade.detectMultiScale(gray, 1.5, 20)

        # Select which emoji to use
        over = None
        if len(smiles) > 0:
            over = emojis["smiley"]
        else:
            over = emojis["yeehaw"]
        
        # Create smiley just for this detection
        this_detection_over = cv2.resize(over, (h, w))

        # Convert all transparent pixels to pixels from main imagae
        add_x, add_y = 0, 0
        for line in this_detection_over:
            add_x = 0
            add_y += 1
            for pixel in line:
                add_x += 1
                if pixel[3] == 0:
                    this_detection_over[add_y - 1, add_x - 1] = result_image[y + add_y - 1, x + add_x - 1]

        # Add emoji over resulting image
        result_image[y:y+h, x:x+w] = this_detection_over

    return result_image

def blur_faces(image, faces):
    result_image = image.copy()

    # Loop through all faces found
    for (x, y, w, h) in faces:
        # Get the face pixels from starting image
        sub_face = image[y:y+h, x:x+w]

        # apply a gaussian blur on this new recangle image
        sub_face = cv2.GaussianBlur(sub_face, (31, 31), 40)

        # merge this blurry rectangle to our final image
        result_image[y:y+sub_face.shape[0], x:x+sub_face.shape[1]] = sub_face

    return result_image

def face_detect(cascades, emojis, frame, mode):
    # Copy the frame to image
    image  = frame.copy()
    # Coonvert to bgra
    image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # Convert it to gray
    gray = cv2.cvtColor(image_bgra, cv2.COLOR_BGR2GRAY)

    # Equalize Hist
    gray = cv2.equalizeHist(gray)

    # Detect frontal faces
    faces = cascades["frontalFaceCascade"].detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(10, 10),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    # If no frontal faces detected, try to get profile faces
    if len(faces) == 0:
        # Get profile faces
        faces = cascades["profileFaceCascade"].detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(10, 10),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        # If no profile faces detected either, just return the frame
        if len(faces) == 0:
            return frame
    
    # Blur faces
    if mode == "blur":
        frame_to_display = blur_faces(image, faces)

    # Smiley over faces
    if mode == "emoji":
        frame_to_display = image_over(cascades["smileFaceCascade"], emojis, image, faces)

    if mode == "rect":
        frame_to_display = draw_rect(image, faces)

    return frame_to_display

def process_frames(args):
    # Read the cascade haar file
    cascades = {}

    # Read emojis
    emojis = {}

    if args.mode == "emoji":
        # Add things needed for emojis (haar and emoji PNGs)
        cascades["smileFaceCascade"] = cv2.CascadeClassifier("./cascades/haarcascade_smile.xml")
        emojis["smiley"] = cv2.imread("./images/smile.png", cv2.IMREAD_UNCHANGED)
        emojis["yeehaw"] = cv2.imread("./images/yeehaw.png", cv2.IMREAD_UNCHANGED)
    cascades["profileFaceCascade"] = cv2.CascadeClassifier("./cascades/haarcascade_profileface.xml")
    cascades["frontalFaceCascade"] = cv2.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")


    # Capture video
    cap = cv2.VideoCapture(0)

    # While there is video
    while(cap.isOpened()):
        # Capture ret and frame
        ret, frame = cap.read()

        # If frame is None break the loop
        # OpenCV loads empty frames near the end of video
        if frame is not None:
            # Detect faces in frame
            processed_frame = face_detect(cascades, emojis, frame, args.mode)

            # Resize the frame to fit the output file format
            processed_frame = cv2.resize(processed_frame, (640, 480))

            # Show the frame in window (useful for real-time detection)
            cv2.imshow('frame',processed_frame)

        # If q is pressed, cancel the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()

    # add CLI arguments
    parser.add_argument('-m', '--mode', action="store", type=str,
                        help="blur, rect, emoji", required=True)

    # parse arguments
    args = parser.parse_args()

    process_frames(args)