import cv2
 
def play_videoFile(filePath,mirror=False):
 
    cap = cv2.VideoCapture(filePath)
    cv2.namedWindow('Video Life2Coding',cv2.WINDOW_AUTOSIZE)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    while True:

        ret_val, frame = cap.read()
        fgmask = fgbg.apply(frame)

        if mirror:
            frame = cv2.flip(frame, 1)

        cv2.imshow('Video Life2Coding', fgmask)
 
        if cv2.waitKey(20) == 27:
            break  # esc to quit
 
    cv2.destroyAllWindows()
 
def main():
    play_videoFile('C:/Users/Nick/Documents/GitHub/OpenCV_Python_Practice/highway1.mp4',mirror=False)
 
if __name__ == '__main__':
    main()
