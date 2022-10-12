import cv2
import datetime

cap = cv2.VideoCapture('C:/detect2/samplevideo.mp4')

# Modify the frame resolution. The default frame resolution is system dependent. 
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


# Export the video from each run 
out = cv2.VideoWriter(f'C:/CV/test{datetime.date.today()}.mp4',cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (frame_width,frame_height))

while True:
    # Capture each frame of the video
    ret, frame = cap.read()
    if ret:    

        # Apply gaussian blurring to the frames
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # Apply dilation to the frames
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply laplacian edge detection to convert frames to binary format
        frame = cv2.Laplacian(src=frame, ddepth=cv2.CV_8U, ksize=3)

        # Begin detection
        # Feed the pretrained model to the classifier
        car_cascade = cv2.CascadeClassifier('C:/detect2/car.xml')
        # Detect objects of different sizes from your input video and return a list of rectangles
        cars = car_cascade.detectMultiScale(gray, 1.1, 1)
        # Read the list of rectangles to draw rectangle boundaries around the cars in each frame
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)


        # Display video 
        cv2.imshow("frame", frame)

        # Save video frame
        out.write(frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

        
# Release video capture
cap.release()

# Close the video window
cv2.destroyAllWindows()
   

        
