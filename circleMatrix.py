import cv2
import numpy as np
from PIL import Image

# Load img
image = cv2.imread('image5.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold image to isolate the white circle
_, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Store the center and radius of circle
circle_center = None
circle_radius = None

for contour in contours:
    # Find smallest circle for each contour
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)

    # If this the biggest circle (assume the circle that we're looking for is the biggest one)
    if circle_radius is None or radius > circle_radius:
        circle_center = center
        circle_radius = radius

print("Center of the circle: ", circle_center)
print("Radius of the circle: ", circle_radius)

# Crop img to circle
if circle_center and circle_radius:
    x, y = circle_center
    left = max(0, x - circle_radius)
    right = min(image.shape[1], x + circle_radius)
    top = max(0, y - circle_radius)
    bottom = min(image.shape[0], y + circle_radius)

    cropped_image = image[top:bottom, left:right]

    cv2.imshow('Cropped to Circle', cropped_image)
    cv2.waitKey(0)

    # Use the radius to calculate the grid size
    cell_size = circle_radius * 2

    # Calculate the number of cells in each dimension
    num_cells_x = image.shape[1] // cell_size
    num_cells_y = image.shape[0] // cell_size

    grid_images = []

    # Divide the image into a grid and store each grid cell image
    for i in range(num_cells_y):
        for j in range(num_cells_x):
            # Calculate the top left corner of the current cell
            x_start = j * cell_size
            y_start = i * cell_size

            # Crop the image to the current cell
            cropped_image = image[y_start:y_start + cell_size, x_start:x_start + cell_size]
            grid_images.append(cropped_image)


    for i, img in enumerate(grid_images):
        cv2.imwrite("gridCell-" + i.__str__() + ".jpg", img)


    # Draw grid lines on image
    image_with_grid = image.copy()

    # Draw vertical grid lines
    for i in range(num_cells_x + 1):
        x_start = i * cell_size
        cv2.line(image_with_grid, (x_start, 0), (x_start, image.shape[0]), (0, 255, 0), 2)

    # Draw horizontal grid lines
    for j in range(num_cells_y + 1):
        y_start = j * cell_size
        cv2.line(image_with_grid, (0, y_start), (image.shape[1], y_start), (0, 255, 0), 2)

    # Resize grid image so it can actually appear on screen
    image_with_grid = cv2.resize(image_with_grid, (image_with_grid.shape[1] // 2, image_with_grid.shape[0] // 2), interpolation=cv2.INTER_AREA)

    # Display the image with the grid
    cv2.imshow('Image with Grid', image_with_grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
