import cv2
import threading
from deepface import DeepFace
import time
from contextlib import suppress

# create a video capture object and specify the default device camera to record videos
# cv2.CAP_DSHOW is a flag that tell the DirectShow API to do a system call and open the camera
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# specify frame width and height
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 600)

# this counter will be used to specify the number of frames the loop will iterate after it again
# that depend on your camera maybe it can process 60 frame per second ... etc
counter = 0
# boolean variable to indicate if a match has been found between my image and the captured video frame
matched = False
# boolean flag to indicate if the user logged or not
logged = False
# read colored image as NumPy array with pixel values representing the color of each pixel in the image.
img = cv2.imread('me.jpg')

def isMatched(frm):
    global matched
    # here iam using suppress because verify() return an error if the face is not matched and i don't care about error
    with suppress(Exception):
        # verify() function detect both faces from referance image and the captured frame by using pre-trained machine learning detector model
        # and return dictionary with key called 'verified' which is equal to true or false.
        # use copy to avoid access denied error from the original image file
        if DeepFace.verify(frm, img.copy())['verified']:
            matched = True

while True:
    # reads the next frame from the video capture device and returns two values: a boolean value indicating whether the frame was successfully read (is_frame_read)
    # and the frame itself (frm), represented as a NumPy array.
    is_frame_read,frm = capture.read()

    if is_frame_read:
        # update video frame every 30 frames
        if counter % 30 == 0:
            # use threading to make sure that the function that compare is work dependently in its own thread
            # and parallel with the main thread which is the video capturing
            threading.Thread(target=isMatched,args=(frm.copy(),)).start()
        counter += 1

        # write matched or not directly on the frame
        if matched:
            cv2.putText(frm,"matched",(120,450),cv2.FONT_HERSHEY_DUPLEX,2,(0,255,0),3)
            logged = True
        else:
            cv2.putText(frm,"not matched",(120,450),cv2.FONT_HERSHEY_DUPLEX,2,(0,0,255),3)

        #  displays/update the captured frame in a window on the screen
        cv2.imshow("is this me?",frm)

    # cv2.waitKey(n milliseconds) for user input before updating the frame
    key = cv2.waitKey(3)
    # ord(char) return the ascii code of the given char
    if key == ord('q') :
        break

    # if the user logged in print msg and break
    if logged:
        # sleep to let the frame shown for a while before it break
        time.sleep(1.5)
        print("you are logged in successfully,done !!!")
        break

cv2.destroyAllWindows()