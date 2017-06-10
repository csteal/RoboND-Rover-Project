import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    terrain_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    terrain_select[above_thresh] = 1

    # Convert RGB image to BGR image
    img_bgr = np.zeros_like(img)
    img_bgr[:,:,0] = img[:,:,2]
    img_bgr[:,:,1] = img[:,:,1]
    img_bgr[:,:,2] = img[:,:,0]
    # Convert BGR to HSV
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Define range of yellow color in HSV
    lower_yellow = np.array([20,100,100])
    upper_yellow = np.array([40,255,255])
    # Threshold the HSV image to get only rock
    rock_select = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

    # Return the binary images
    return cv2.bitwise_not(terrain_select), rock_select, terrain_select

# Define a function to convert to rover-centric coordinates
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the
    # center bottom of the image.
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle)
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to apply a rotation to pixel positions
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    # Apply a rotation
    xpix_rotated = xpix * np.cos(yaw_rad) - ypix * np.sin(yaw_rad)
    ypix_rotated = xpix * np.sin(yaw_rad) + ypix * np.cos(yaw_rad)
    # Return the result
    return xpix_rotated, ypix_rotated

# Define a function to perform a translation
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale):
    # Apply a scaling and a translation
    xpix_translated = xpos + xpix_rot / scale
    ypix_translated = ypos + ypix_rot / scale
    # Return the result
    return xpix_translated, ypix_translated

# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size=200, scale=10):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image

    return warped


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()

    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    # the following copied from Rover_Project_Test_Notebook
    dst_size = 5
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - bottom_offset],
                  [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - bottom_offset],
                  [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset],
                  [Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset]])

    # 2) Apply perspective transform
    warped = perspect_transform(Rover.img, source, destination)

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    obs_thresh, rock_thresh, nav_thresh = thresh(warped)

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    Rover.vision_image = np.zeros((160, 320, 3), dtype=np.float)
    Rover.vision_image[obs_thresh.nonzero()[0],obs_thresh.nonzero()[1],0] = 255
    Rover.vision_image[rock_thresh.nonzero()[0],rock_thresh.nonzero()[1],1] = 255
    Rover.vision_image[nav_thresh.nonzero()[0],nav_thresh.nonzero()[1],2] = 255

    # 5) Convert map image pixel values to rover-centric coords
    obstacle_x, obstacle_y = rover_coords(obs_thresh)
    rock_x, rock_y = rover_coords(rock_thresh)
    navigable_x, navigable_y = rover_coords(nav_thresh)
    # 6) Convert rover-centric pixel values to world coordinates
    obstacle_x_world, obstacle_y_world = pix_to_world(obstacle_x,
                                                        obstacle_y,
                                                        Rover.pos[0],
                                                        Rover.pos[1],
                                                        Rover.yaw)
    rock_x_world, rock_y_world = pix_to_world(rock_x,
                                                rock_y,
                                                Rover.pos[0],
                                                Rover.pos[1],
                                                Rover.yaw)
    navigable_x_world, navigable_y_world = pix_to_world(navigable_x,
                                                        navigable_y,
                                                        Rover.pos[0],
                                                        Rover.pos[1],
                                                        Rover.yaw)
    # 7) Update Rover worldmap (to be displayed on right side of screen)
    Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
    Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
    Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1

    # 8) Convert rover-centric pixel positions to polar coordinates
    rover_centric_pixel_distances, rover_centric_angles = to_polar_coords(navigable_x, navigable_y)
    rover_centric_rock_distances, rover_centric_rock_angles = to_polar_coords(rock_x, rock_y)
    # Update Rover pixel distances and angles
    Rover.nav_dists = rover_centric_pixel_distances
    Rover.nav_angles = rover_centric_angles
    #Rover.nav_dists, Rover.nav_angles = to_polar_coords(navigable_x, navigable_y)
    Rover.rock_dists = rover_centric_rock_distances
    Rover.rock_angles = rover_centric_rock_angles



    return Rover
