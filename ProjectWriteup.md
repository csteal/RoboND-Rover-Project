## Project: Search and Sample Return

### Notebook Analysis
#### Obstacle identification
To identify obstacles I took the inverse of the navigable terrain image. I did this using `cv2.bitwise_not` in the `thresh()`
#### Rock identification
To identify rocks I used two functions from [OpenCV](http://opencv.org), namely `cv2.cvtColor()` and `cv2.inRange()`. I followed along with a [changing colorspaces](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html)
