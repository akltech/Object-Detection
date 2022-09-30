import cv2
import numpy as np
import datetime

cap = cv2.VideoCapture('C:/detect2/samplevideo.mp4')

# Obtain the default resolutions of the frame. The frame resolutions are system dependent. 
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


# Exports the video from each run 
out = cv2.VideoWriter(f'C:/detect2/newvideos/output_video{datetime.date.today()}.mp4',cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (frame_width,frame_height))

while True:
    # Capture each frame of the video
    ret, frame = cap.read()
    if ret:    

        # Apply gaussian blurring to the frames
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # Apply dilation to the frames
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply laplacian edge detection (binary format)
        frame = cv2.Laplacian(src=frame, ddepth=cv2.CV_8U, ksize=3)

        # Begin detection
        # The classifier reads the pretrained model
        car_cascade = cv2.CascadeClassifier('C:/detect2/car.xml')
        # Detects objects of different sizes from our input video and outputs list of rectangles
        cars = car_cascade.detectMultiScale(gray, 1.1, 1)
        # Reads the list of rectangles to draw rectangle boundaries in each frame
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)


        # Displays video 
        cv2.imshow("frame", frame)

        #saves video frame
        out.write(frame)
        
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

        
# Release video capture
cap.release()

# Close the video window
cv2.destroyAllWindows()
   

        

