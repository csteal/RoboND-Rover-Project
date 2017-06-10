## Project: Search and Sample Return

### Notebook Analysis
#### Obstacle identification
To identify obstacles I took the inverse of the "navigable terrain" image. I did this using `cv2.bitwise_not` in the `color_thresh()`
#### Rock identification
To identify rocks I used two functions from [OpenCV](http://opencv.org), namely `cv2.cvtColor()` and `cv2.inRange()`. I followed along with a [changing colorspaces](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html) to find the yellow regions (rocks) in the images. These additions were made in the `color_thresh()`

#### `process_image()`
The `process_image()` was modified using the following steps:
* Apply perspective transform to input image using `perspect_transform()`
* Apply threshold to image to identify navigable terrain, obstacles, and rock samples using `color_thresh()`
* Convert threshold image pixel values to rover-centric coords using `rover_coords()`
* Covert rover-centric pixel values to world coords using `pix_to_world()`
* Update the worldmap to include obstacle positions, terrain positions and rock sample positions
* The code for the mosaic image was already provided and no additions were necessary

Video output is provided in `..\output`

### Autonomous Navigation and mapping
#### `perception_step()`
All of the functions from the notebook analysis where copied over. The difference for the file was that an instance of the class `Rover_State()` was passed into `perception_step()`. Some minor changes allowed the image stored in `Rover.image` to be processed

#### `decision_step()`
The one major change I made to the code was the addition of logic corresponding with a rock detection while traveling. The variables `Rover.rock_angles` and `Rover.rock_dists` are used to navigate towards a rock sample.

**Note: Simulator was run with 640 x 480 screen resolution, 'good' graphics quality and FPS of 45
