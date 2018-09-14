# Imports
import cv2
import numpy as np

# Video Playing Function
def play_videoFile(filePath):

    # Capture the Video from the given path and window creation
    cap = cv2.VideoCapture(filePath)
    cv2.namedWindow('Video', cv2.WINDOW_AUTOSIZE)

    #Low Video Resolution
    cap.set(3, 100)
    cap.set(4, 100)

    # Background Subtractor creation 
    fgbg = cv2.createBackgroundSubtractorMOG2()

    # Setup SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()
    
    # Change thresholds
    params.minThreshold = 1
    params.maxThreshold = 10000
    
    # Filter by Area
    params.filterByArea = True
    params.minArea = 600
    
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0
    
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0
    
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    while True:

        # Frame Reading
        ret_val, frame = cap.read()

        # Applying Mask
        fgmask = fgbg.apply(frame)
        
        # Applying Blod Detector to find connected pixels
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(fgmask)

        # Drawing points at the detected keypoints
        points = cv2.drawKeypoints(fgmask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Display Frames
        cv2.imshow('Video', points)

        # Break Functionality 
        if cv2.waitKey(50) == 27:
            break  # esc to quit

    cv2.destroyAllWindows()

# Python Main
def main():
    play_videoFile('C:/Users/Nick/Documents/GitHub/OpenCV_Python_Practice/highway3.mp4')

# Starting If
if __name__ == '__main__':
    main()
