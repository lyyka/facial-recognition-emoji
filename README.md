# facial-recognition-emoji
App made using Python and opencv-python that detects faces from live web camera feed and can frame them, blur them, or even put emojis over them!

# Requires
This app requires opencv-python and pillow packages both to be installed in order to work.<br/>
You can install those in your environment using pip.<br/>
pip install opencv-python<br/>
pip install pillow

# Arguments
The video file has just a **mode** (-m or --mode) argument. It specifies what do you want to do with faces when detected.<br/>
blur (to blur faces)<br/>
rect (to frame the faces)<br/>
emoji (to put emoji over faces)
<br/>
The image file has two arguments, **mode** and **image**. Mode is the same as above, and image just specifies path to the image relative to the video file which you want to use to detect faces on