# Create an OpenCV Program for Vehicle Detection 

This tutorial provides a hands-on, step-by-step introduction to video object detection with OpenCV-Python, the Python API of OpenCV. To create your own vehicle detection program, complete the following steps.

## Tasks

  * [Before you Begin](#before-you-begin)

  * [Step 1: Read, display, and write a video](#step-1-read-display-and-write-a-video)

  * [Step 2: Apply image preprocessing techniques](#step-2-apply-image-preprocessing-techniques)

  * [Step 3: Use the Haar Cascade Classifier to detect vehicles](#step-3-use-the-haar-cascade-classifier-to-detect-vehicles)

  * [Common Errors](#common-errors)


## Requirements

This tutorial uses the computer vision library OpenCV, if you do not already have OpenCV installed please [visit here](https://medium.com/@pranav.keyboard/installing-opencv-for-python-on-windows-using-anaconda-or-winpython-f24dd5c895eb) for Window's installation. 
Familiarity with image processing terms, such as noise reduction, is helpful but not required. Experience with machine learning algorithms is also not required. 


## Before you begin 

* If you installed OpenCV on Windows with Anaconda, open the anaconda prompt and run the command ```pip install opencv-contrib-python```. OpenCV-contrib is an additional module that contains Haar Cascades classifiers, the machine learning detection algorithms that you will use in your program. 

* Download the samplevideo.mp4 and car.xml files from the program files [here](https://github.com/akltech/Vehicle-Detection/blob/94c946f07dfba7fcd958893d59c074fbe26fe91a/Program%20Files) and extract the contents into your code directory. The program file contains a sample input video and a pre-trained classifier for cars. Training refers to feeding the machine learning algorithm data so that it can learn to detect specific objects. The XML file includes features classified as a car or non-car. 

## Step 1: Read, display and write a video.

OpenCV's HighGUI API provides methods to communicate with your computer hardware and file system and open and interact with a GUI (Graphical User Interface). A GUI allows you to interact with a computer system through visual graphics, such as the file icons on a Windows OS. With the HighGUI module, you can read and write videos into your file system and view them in real time in OpenCV's High-level GUI.

To import, view, and export a video, follow these steps:

  1. Import the dependencies: ```cv2``` and ```DateTime```

  2. OpenCV’s HighGUI allows you to load a video from your file system by calling the ```VideoCapture()``` method. You can pass a parameter to ```VideoCapture()``` to tell it to capture a file from your computer or through a webcam. Though in this tutorial you will enter a file path or filename as an argument, you can typically pass in zero as a parameter to retrieve your computer’s webcam feed–for example, ```VideoCapture(0)```. 
  <br></p>
The following code sample shows how to use the ```VideoCapture()``` method to load a video from a file. For information about other methods, see [cv Namespace Reference.](https://docs.opencv.org/4.x/d2/d75/namespacecv.html)
<br></p>
    ```
    cap = cv2.VideoCapture('video path')
    ```
  3. Set the output video's frame size by calling the ``` get()``` method. To obtain the default resolution of the frames in a video stream, you can pass an enumerator as a parameter to the ``` get()``` method--for example, ```CAP_PROP_FRAME_WIDTH```. The default resolutions of your system will apply. 
    <br></p>
Alternatively, you can modify your output video's frame size by passing a number as a parameter to the ```get()``` method, as follows in the code sample below. For more information about other enumerators, see [Enumerations.](https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html) 
  <br></p>
    ```
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    ```
  4. To create a video file, make an instance of the ``` VideoWriter()``` class. The first parameter is the filename or the path to the output video file. Next, you must provide the fourcc codec, a four-character code. A codec is a hardware or software that compresses or decompresses a digital video–for example, you can compress an MP4 with the MPEG4 codec```mp4v```. This way, the output video frames will be compressed to a smaller file size suitable for viewing from your directory. To view a list of fourcc codes, see [Video FOURCCs.](https://learn.microsoft.com/en-us/windows/win32/medfound/video-fourccs)
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

**To apply image preprocessing techniques**

  1. Use the function ```GaussianBlur()``` to apply the Gaussian blur. Enter your source image, and width and height dimensions for the kernel size. The kernel size is used to determine the intensity of the applied effects. The blurring effect will increase or decrease with the kernel dimensions.
  ```frame = cv2.GaussianBlur(frame, (5, 5), 0)```
  
  2. Apply dilation to accentuate features and further blur the image with ```Dilate()```.
  ```cv2.Dilate(frame, (5,5), 2)```
  
  3. ```cvtColor()``` method changes the color space of an image. You can convert each of the frames to a gray scale by using the space transformation technique ```BGR GRAY```
  ```gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)```

  4. (Optional) Most computer vision programs convert the image to binary format before feeding it to a machine learning algorithm. A binary image consists of pixels with only one or two colors, usually black and white. In computer vision, edge detection is one of the standard methods for converting an image to binary format by producing the edges and curves of an image in white on a black background. The easiest and most effective way to apply edge detection to a video is by using Laplacian edge detection, a method in the OpenCV library. An image's Laplacian highlights areas where the intensity changes rapidly, making Laplacian edge detection ideal for converting video to binary.
  Use the ```Laplacian``` method to apply edge detection to your video.
  ```
  frame = cv2.Laplacian(src=frame, ddepth=cv2.CV_8U, ksize=3)
  ```
  <br />
<p align="center">
  <img width="440" height="430" src="Laplacian_edge_detection.jpg">
</p>
<br />
<p align="center">
Laplacian Edge Detection of a highway. 
</p>
<br />

## Step 3: Use the Haar Cascade Classifier to detect vehicles

After preprocessing the frames, use pre-trained Haar cascade models to detect cars in a video.

**To run the detection algorithm**

  1. Read the necessary XML file, car.xml, using the ```CascadeClassifier``` method. 
  ```
  # the classifier reads the pretrained model
        car_cascade = cv2.CascadeClassifier('car.xml')
  ```
  
  2. Use the ```detectMultiScale()``` to detect the vehicles. ```detectMultiScale()``` detects objects of different sizes in our input video and returns the detected objects as a list of rectangles. 
  The ```rectangle()`` method reads from list of rectangles that you processed from our video to draw rectangles around the cars.
  ```
        # detects objects of different sizes from our input video and outputs list of rectangles
        cars = car_cascade.detectMultiScale(gray, 1.1, 1)
        
        # reads the list of rectangles to draw rectangle boundaries in each frame
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
  ```
 
Access the complete code [here.](https://github.com/akltech/Vehicle-Detection/blob/b836e9383ba34d36bf3994b9ad38f19355a84dee/vehicle_detection_model.py)

##  Common Errors 

This section lists errors common to the openCV-Python API.

<br />

**The output video is not playing**

  > This item was encoded in a format that's not supported.

One possible solution is to download the VLC media player for Windows [here](https://www.videolan.org/vlc/download-windows.html).
Then, right click on the video and go to ```Properties```, ```Change...```, and select ```VLC Media Player```.

<br />

**Cascaade Classifier is not performing as expected**

  > SystemError: <class 'cv2.CascadeClassifier'> returned a result with an error set
  
Ensure that you have ```opencv-contrib-python``` installed. Return to [Before you Begin](#before-you-begin) for installation instructions. 


<br />
<br />
<p align="center">
  <img width="450" height="440" src="https://github.com/akltech/Vehicle-Detection/blob/d487f0d26100899e4b5cb21a5bea76590978627d/detectionscreenshot.JPG">
</p>



