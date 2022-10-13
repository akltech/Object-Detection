# Create an OpenCV Program for Vehicle Detection 

This tutorial provides a step-by-step introduction to object detection with video using OpenCV-Python, the Python API of OpenCV, a computer vision library. To create a vehicle detection program, complete the following steps:

## Objectives

  * [Before you Begin](#before-you-begin)

  * [Step 1: Read, display, and write a video](#step-1-read-display-and-write-a-video)

  * [Step 2: Apply image preprocessing techniques](#step-2-apply-image-preprocessing-techniques)

  * [Step 3: Use the Haar Cascade Classifier to detect vehicles](#step-3-use-the-haar-cascade-classifier-to-detect-vehicles)

  * [Common Errors](#common-errors)


## Requirements

OpenCV and Python are required to run the code sample locally. If you are on Windows and do not already have OpenCV installed see <a href="https://medium.com/@pranav.keyboard/installing-opencv-for-python-on-windows-using-anaconda-or-winpython-f24dd5c895eb" target="_blank">Installing OpenCV for Python.</a>
Familiarity with computer vision and machine learning terms is helpful but not required. 


## Before you begin 

* Install OpenCV-contrib. OpenCV-contrib contains Haar Cascades classifiers, or machine learning detection algorithms that you will use in your program. When OpenCV-Python is installed with Anaconda, the Haar Cascade classifiers are typically missing. 
  <br></p>
To install the extra module, open the Anaconda prompt and use the ```pip``` <a href="https://pypi.org/project/opencv-contrib-python/" target="_blank">command:</a>
    ```
    pip install opencv-contrib-python
    ```

*  Download the ```samplevideo.mp4``` file and the ```car.xml``` file from <a href="https://github.com/akltech/Vehicle-Detection/blob/94c946f07dfba7fcd958893d59c074fbe26fe91a/Program%20Files" target="_blank">Program Files</a> into your code directory. The ```samplevideo.mp4``` file contains a sample input video. 
<br></p>
The XML file contains a pre-trained classifier for cars. Training refers to feeding the machine learning algorithm data so that it can learn to detect specific objects. ```car.xml``` includes features classified as a car or non-car. This way, the algorithm training for vehicles is already complete. 

## Step 1: Read, display and write a video.

OpenCV's HighGUI API provides methods to access your computer hardware and file system and display video streams in the High-level GUI (Graphical User Interface). A GUI allows you to interact with a computer system through visual graphics, such as clickable file icons on a Windows OS. With the HighGUI module, you can read and write videos into a given directory and view them in real time.

To load, view, and write a video, follow these steps:

  1. Import the dependencies: ```cv2``` and ```DateTime```

  2. OpenCV’s HighGUI allows you to load a video from your file system by calling the ```VideoCapture()``` method. You can pass a parameter to ```VideoCapture()``` to tell it to capture a file from your computer or through a webcam. Though in this tutorial you will enter a file path or filename as an argument, you can typically pass in zero as a parameter to retrieve your computer’s webcam feed–for example, ```VideoCapture(0)```. 
  <br></p>
The following code sample shows how to use the ```VideoCapture()``` method to load a video from a file. For information about other methods, see <a href="https://docs.opencv.org/4.x/d2/d75/namespacecv.html" target="_blank">cv Namespace Reference.</a>
<br></p>
    ```
    cap = cv2.VideoCapture('video path')
    ```
  3. Set the output video's frame size by calling the ``` get()``` method. To obtain the default resolution of the frames in a video stream, you can pass an enumerator as a parameter to the ``` get()``` method--for example, ```CAP_PROP_FRAME_WIDTH```. The default resolutions of your system will apply. 
    <br></p>
Alternatively, you can modify your output video's frame size by passing a number as a parameter to the ```get()``` method, as follows in the code sample below. For more information about other enumerators, see <a href="https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html" target="_blank">Enumerations.</a>
  <br></p>
    ```
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    ```
  4. To create a video file, make an instance of the ``` VideoWriter()``` class. The first parameter is the filename or the path to the output video file. Next, you must provide the fourcc codec, a four-character code. A codec is a hardware or software that compresses or decompresses a digital video–for example, you can compress an MP4 with the MPEG4 codec```mp4v```. This way, the output video frames will be compressed to a smaller file size suitable for viewing from your directory. To view a list of fourcc codes, see <a href="https://learn.microsoft.com/en-us/windows/win32/medfound/video-fourccs" target="_blank">Video FOURCCs.</a>
  <br></p>
The final parameters of this method are, respectively, the output video's FPS (frames per second) and the size of the output video. Setting a framerate that is too fast or slow will interfere with your detection algorithm's accuracy. A good starting point for the FPS is ten seconds. The following code sample shows how to call the ``` VideoWriter()``` method to write a video file in a given directory with today’s date:
  <br></p>
    ```
    out =    cv2.VideoWriter(f'C:/detect2/test{datetime.date.today()}.mp4',
    cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10,  (frame_width,frame_height)) 
    ```
  5. Initialize a loop with a return value to indicate if you have captured the video frames. To capture each frame, invoke the ```read()``` function:
    <br></p>
    ```
    while True:
    ret, frame = cap.read()
    if ret:    
    ```
  6. To display the video frames in the HighGUI window, call the ```imshow()``` method: 
  <br></p>
    ```
    cv2.imshow('frame', frame)
    ```
  7. Pass the frames into the video file by calling the ```write()``` function: 
  <br></p>
    ```
    out.write(frame)
    ```
  8. Pass the ```waitkey()``` method a parameter to tell it the number of milliseconds to wait for a pressed key to close the window. The ```waitKey()``` method is when OpenCV's HighGUI event messages are processed–for example, ```waitKey(20)``` waits up to twenty milliseconds to stop processing the GUI events. Regardless of the timeout value, "'waitKey()"' returns instantly with key input. The code sample that follows shows how to call the ```waitKey()``` method to exit a loop with a keyboard event after twenty-five seconds:
  <br></p>
    ```
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    ```
  9. Use the ```release()``` function to close all frames. This function closes the video capture object. 
  <br></p>
    ```
    cap.release()
    ```
  10. To destroy the GUI window, call the ```destroyAllWindows()``` method. Usually, the GUI window will close when the program terminates, but if you are executing it in Python from the terminal rather than a script, it can remain open until you quit Python. Calling ```destroyAllWindows()```  is a good practice to prevent an open GUI window from lingering.
  <br></p>
    ```
    cv2.destroyAllWindows()
    ```
    
## Step 2: Apply image preprocessing techniques

To better detect objects in an image, you must process the image before feeding it to the detection algorithm or the Haar Cascades classifier. Applying techniques with OpenCV, such as gaussian blurring and dilation, reduces the algorithm's complexity. You will begin by applying a gaussian blur to reduce the noise level and smooth the image. For information about image preprocessing functions in OpenCV, see <a href="https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html" target="_blank">Image Filtering.</a>

**Apply image preprocessing techniques:**

  1. Use the function ```GaussianBlur()``` to apply the Gaussian blur. Enter your source image, and width and height dimensions for the kernel size. The kernel size is used to determine the intensity of the applied effects. The blurring effect will increase or decrease with the kernel dimensions.
  ```frame = cv2.GaussianBlur(frame, (5, 5), 0)```
  
  2. Apply dilation to accentuate features and further blur the image with ```Dilate()```.
  ```cv2.Dilate(frame, (5,5), 2)```
  
  3. ```cvtColor()``` method changes the color space of an image. You can convert each of the frames to a gray scale by using the space transformation technique ```BGR GRAY```
  ```gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)```

  4. Optional: Convert the video to binary format by calling the ```Laplacian()``` method. A binary image usually consists of only two pixels, usually black and white. This way, the computer can more accurately identify the objects. 
   <br></p>
The ```Laplacian()``` method is an edge detection algorithm that returns a binary image. Sudden changes in pixel intensity result in black and white egde pixels. An image's Laplacian highlights areas where the pixel intensity changes rapidly, making it an excellent method to call for identifying edges in a video. For more information about ```Laplacian()```, see <a href="https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gad78703e4c8fe703d479c1860d76429e6">Image Filtering.</a>
  <br></p>
To apply Laplacian edge detection, modify the following code sample:
    <br></p>
    ```
    frame = cv2.Laplacian(src=frame, ddepth=cv2.CV_8U, ksize=3)
    ```
  <br />
  <img width="440" height="430" src="Laplacian_edge_detection.jpg">
</p>
<br />
 

## Step 3: Use the Haar Cascade Classifier to detect vehicles

**Run the vehicle detection algorithm:**

  1. To tell the classifier to read the pre-trained model for cars, call the ```CascadeClassifier()``` method and pass in the ```car.xml``` file as a parameter.  In the following code sample, the pre-trained model for cars is read using the ```CascadeClassifier()``` method.
    <br></p>
    ```
    car_cascade = cv2.CascadeClassifier('car.xml')
    ```
  
  2. To detect objects of different sizes in the input video, call the ```detectMultiScale()``` function.
  
  3. Call the ```rectangle()`` function to to read from the list of rectangles to draw boundary boxes around the cars.
    <br></p>
     ```
     cars = car_cascade.detectMultiScale(gray, 1.1, 1)
        
     for (x, y, w, h) in cars:
         cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
     ```
 
You can access the complete code sample <a href="https://github.com/akltech/Vehicle-Detection/blob/b836e9383ba34d36bf3994b9ad38f19355a84dee/vehicle_detection_model.py6">on Github.</a>

##  Common Errors 

This section lists errors common to the openCV-Python API.

<br />

**This item was encoded in a format that's not supported**

One possible solution is to download the <a href="https://www.videolan.org/vlc/download-windows.html" target="_blank">VLC media player for Windows.</a>
Then, right click on the video and go to ```Properties```, ```Change...```, and select ```VLC Media Player```

<br />

**SystemError: <class 'cv2.CascadeClassifier'> returned a result with an error set***
  
Ensure that you have ```opencv-contrib-python``` installed. Return to [Before you Begin](#before-you-begin) for installation instructions. 





