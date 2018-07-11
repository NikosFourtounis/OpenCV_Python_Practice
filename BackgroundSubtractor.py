import cv2
import numpy as np

def play_videoFile(filePath, mirror=False):

    cap = cv2.VideoCapture(filePath)
    cv2.namedWindow('Video', cv2.WINDOW_AUTOSIZE)
    fgbg = cv2.createBackgroundSubtractorMOG2()

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    
    # Change thresholds
    params.minThreshold = 1
    params.maxThreshold = 2000
    
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 1000
    
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

        ret_val, frame = cap.read()
        fgmask = fgbg.apply(frame)
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(fgmask)
        points = cv2.drawKeypoints(fgmask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        if mirror:
            frame = cv2.flip(frame, 1)

        cv2.imshow('Video', points)

        if cv2.waitKey(50) == 27:
            break  # esc to quit

    cv2.destroyAllWindows()


def main():
    play_videoFile('C:/Users/Nick/Documents/GitHub/OpenCV_Python_Practice/highway1.mp4', mirror=False)


if __name__ == '__main__':
    main()
