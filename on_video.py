# OpenCV for image recognition
import cv2

# For arguments and CMD
import argparse

# For calculating the avg time needed
from time import time


def draw_rect(image, faces):
    # Copy the image to the result
    result_image = image.copy()

    # For each face, draw the rectangle around it
    for (x, y, w, h) in faces:
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Return the resulting image with rectangles
    return result_image


def image_over(smile_face_cascade, emojis, image, faces):
    # Copt the image to resulting one
    result_image = image.copy()

    # Change colors to BGRA
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2BGRA)

    # Define the gray-scale copy of the resulting image
    gray = cv2.cvtColor(result_image, cv2.COLOR_BGRA2GRAY)
    
    # Loop through detected faces
    for (x, y, w, h) in faces:
        # See if there are smileys to switch emojis
        smiles = smile_face_cascade.detectMultiScale(gray, 1.5, 20)

        # Select which emoji to use
        if len(smiles) > 0:
            over = emojis["smiley"]
        else:
            over = emojis["yeehaw"]
        
        # Create smiley just for this detection
        this_detection_over = cv2.resize(over, (h, w))

        # Convert all transparent pixels to pixels from main image that should be below
        add_x, add_y = 0, 0
        for pixel_row in this_detection_over:
            add_x = 0
            add_y += 1
            for pixel in pixel_row:
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
    image = frame.copy()

    # Convert it to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Equalize Hist
    gray = cv2.equalizeHist(gray)

    # Detect frontal faces
    faces = cascades["frontalFaceCascade"].detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(10, 10),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # If no frontal faces detected, try to get profile faces
    if len(faces) == 0:
        # Get profile faces
        faces = cascades["profileFaceCascade"].detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(10, 10),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # If no profile faces detected either, just return the frame
        if len(faces) == 0:
            return frame
    
    # Blur faces
    if mode == "blur":
        frame_to_display = blur_faces(image, faces)

    # Smiley over faces
    elif mode == "emoji":
        frame_to_display = image_over(cascades["smileFaceCascade"], emojis, image, faces)

    # Rectangle over faces
    elif mode == "rect":
        frame_to_display = draw_rect(image, faces)

    # Default option
    else:
        frame_to_display = draw_rect(image, faces)

    return frame_to_display


def process_frames(cmd_args):
    # Read the cascade haar file
    cascades = {}

    # Read emojis
    emojis = {}

    if cmd_args.mode == "emoji":
        # Add things needed for emojis (haar and emoji PNGs)
        cascades["smileFaceCascade"] = cv2.CascadeClassifier("./cascades/haarcascade_smile.xml")
        emojis["smiley"] = cv2.imread("./images/smile.png", cv2.IMREAD_UNCHANGED)
        emojis["yeehaw"] = cv2.imread("./images/yeehaw.png", cv2.IMREAD_UNCHANGED)
    cascades["profileFaceCascade"] = cv2.CascadeClassifier("./cascades/haarcascade_profileface.xml")
    cascades["frontalFaceCascade"] = cv2.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")

    # Capture video
    cap = cv2.VideoCapture(0)

    # total time to process frames
    total_frames_time = 0
    total_frames_num = 0

    # While there is video
    while cap.isOpened():
        # Capture ret and frame
        ret, frame = cap.read()

        # If frame is None break the loop
        # OpenCV loads empty frames near the end of video
        processed_frame = None
        if frame is not None:
            # Get time before the detection
            process_start = time()

            # Detect faces in frame
            processed_frame = face_detect(cascades, emojis, frame, cmd_args.mode)

            # Get time after the q
            process_end = time()

            # Change variables to calculate average time needed
            total_frames_time += (process_end - process_start)
            total_frames_num += 1

            # Resize the frame to fit the output file format
            processed_frame = cv2.resize(processed_frame, (640, 480))

            # Show the frame in window
            cv2.imshow('frame', processed_frame)

        # If s is pressed, save the frame image
        if cv2.waitKey(1) & 0xFF == ord('s'):
            if frame is not None and processed_frame is not None:
                cv2.imwrite("frame_{0}.jpg".format(total_frames_num), processed_frame)

        # If q is pressed, cancel the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("----------")
            print("Average time to process single frame: {0}".format(total_frames_time / total_frames_num))
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()

    # add CLI arguments
    parser.add_argument('-m', '--mode', action="store", type=str,
                        help="Available options: 'blur', 'rect' or 'emoji'", required=True)

    # parse arguments
    args = parser.parse_args()

    process_frames(args)
