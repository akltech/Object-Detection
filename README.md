# Vehicle-Detection

This tutorial provides a hands-on introduction to video object detection with OpenCV-Python, the Python API of OpenCV. To create a vehicle detection program, complete the following steps.

## Tasks

  * [Before you Begin](#before-you-begin)

  * [Step 1: Read and write your input video](#step-1-read-and-write-your-input-video)

  * [Step 2: Apply image preprocessing techniques](#step-2-apply-image-preprocessing-techniques)

  * [Step 3: Use the Haar Cascade Classifier to detect vehicles](#step-3-use-the-haar-cascade-classifier-to-detect-vehicles)

  * [Common Errors](#common-errors)

## Requirements

This tutorial uses the computer vision library openCV, if you do not already have openCV installed please [visit here](https://medium.com/@pranav.keyboard/installing-opencv-for-python-on-windows-using-anaconda-or-winpython-f24dd5c895eb) for Window's installation. 
Familiarity with image processing is helpful but not required.

## Before you begin 

* Open the anaconda prompt and run the command pip install opencv-contrib-python. OpenCV contrib is a specialized module that contains Haar Cascades, and specifically the CascadeClassifier, which will be used to read the pretrained model for cars. The Open-cv contrib module is not included in the opencv installation through anaconda, or pip installation of it.

* Download the detect_demo.zip from here and unzip the contents into your code directory. The file contains a sample input video and an XML file of a pre-trained Haar Cascade classifier for cars.

## Step 1: Read and write your input video

OpenCV has built-in functions for importing and opening the frames of video files. For this tutorial, you will include an input and output path.

**To display and export your video automatically**

  1. Import the following libraries: ```cv2``` ```numpy``` ```datetime```
  
  2. Play the video by from a file by changing the camera index in the ```cv.CascadeClassifier()``` method of the ```cv2``` library to a file name. The following code can be modified for your use:
  ```
  input_video = 'C:/CVdetect/samplevideo.mp4'
  cap = cv2.VideoCapture(input_video)
  ```
  
  3. Obtain the default resolutions of the video frame. The default resolutions depend on your system.
  ```
  frame_width = int(cap.get(3))
  frame_height = int(cap.get(4))
  ```
  
  4. Automatically export the video into a directory using the ```cv.VideoWriter``` method.
  ```
  #writes a new video for each iteration with today's date
  out = cv2.VideoWriter(f'C:/detect/test{datetime.date.today()}.mp4',cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (frame_width,frame_height))
  ```
  
  5. Capture each frame of the video with the cap.read() function. 
  ```
  # loop runs if the video is succesfully played
  while True:
  
    # captures each frame of the video
    ret, frame = cap.read()
    if ret:
   ```
    
## Step 2: Apply image preprocessing techniques

To better detect objects in image, you must process the image before feeding it to the detection algorithm, or the Haar Cascades classifier. Applying techniques with openCV such as gaussian blurring, reducing noise and smoothing the image reduces the complexity of the applied algorithm. You will begin by applying a gaussian blur to reduce the noise level and dilating the image. For more information about image pre-processing functions in openCV, see Image Filtering.

**To apply image preprocessing techniques**

  1. Use the function ```cv.GaussianBlur()``` to apply the Gaussian blur. Enter your source image, and width and height dimensions for the kernel size. The kernel size is used to determine the intensity of the applied effects. The blurring effect will increase or decrease with the kernel dimensions.
  ```frame = cv2.GaussianBlur(frame, (5, 5), 0)```
  
  2. Dilation may be done to accentuate features and further blur the image with ```cv.Dilate()```.
  ```cv2.Dilate(frame, (5,5), 2)```
  
  3. ```cv.cvtColor()``` method changes the color space of an image. You can convert each of the frames to a gray scale by using the space transformation technique ```BGR GRAY```
  ```gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)```

  4. (Optional) Most computer vision programs convert the image to binary format before feeding it to a machine learning algorithm. A binary image consists of pixels with only one or two colors, usually only black and white.Edge detection is commonly used to convert an image to binary format by detecting the edges and curves and displaying them on a dark background.
  Use the ```cv.Laplacian``` method, which applies Laplacian edge detection, to see this result.
  ```frame = cv2.Laplacian(src=frame, ddepth=cv2.CV_8U, ksize=3)```
  
   ![Laplacian Edge Detection](laplcian.JPG)
  
## Step 3: Use the Haar Cascade Classifier to detect vehicles

After preprocessing the frames, use pre-trained Haar cascade models to detect cars in a video.

**To run the detection algorithm**

  1. Read the necessary XML file, car.xml, using the ```cv.CascadeClassifier``` method. 
  ```
  # the classifier reads the pretrained model
        car_cascade = cv2.CascadeClassifier('car.xml')
  ```
  
  2. Use the ```cv.detectMultiScale()``` to detect the vehicles. ```cv.detectMultiScale()``` detects objects of different sizes in our input video and returns the detected objects as a list of rectangles. 
  The ```cv.rectangle()`` method reads from list of rectangles that we processed from our video to draw rectangles around the cars.
  ```
        # detects objects of different sizes from our input video and outputs list of rectangles
        cars = car_cascade.detectMultiScale(gray, 1.1, 1)
        
        # reads the list of rectangles to draw rectangle boundaries in each frame
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
  ```
 
 
  3. Display and stop your detection video with the ```cv.imshow``` and ```cv.waitKey``` methods. ```cv.waitKey``` waits x amount of time in milliseconds for the given time to destroy the window. Pass 0 through the argument for the ```waitKey()``` function to wait for a key press.
  ```
  #displays video 
        cv2.imshow("frame", frame)
        
        #Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
   ```

##  Common Errors 

This section lists errors common to the openCV-Python API.

**The output video is not playing**

  > This item was encoded in a format that's not supported.
You may receive this error when you try to play an output video on Windows. 

One possible solution is to download the VLC media player for Windows [here](https://www.videolan.org/vlc/download-windows.html).
Then, right click on the video and go to ```Properties```, ```Change...```, and select ```VLC Media Player```.


**Cascaade Classifier is not performing as expected**

  > SystemError: <class 'cv2.CascadeClassifier'> returned a result with an error set
  
Ensure that you have ```opencv-contrib-python``` installed. Return to [Before you Begin](#before-you-begin) for installation instructions. 

 ![sample detection](https://imgur.com/LJOZk7X)

