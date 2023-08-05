import cv2
import numpy as np

image_path = r'C:\Users\user\Desktop\OpenCV Task\rectangles.png'
image = cv2.imread(image_path)

def measure_inner_line_lengths(rectangle, image):
    # Calculate the rotated bounding box of the rectangle contour
    rect = cv2.minAreaRect(rectangle)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Create a mask for the rectangle
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [box], -1, (255), thickness=cv2.FILLED)

    # Find edges inside the rectangle
    edges = cv2.Canny(image, threshold1=50, threshold2=150)
    edges_inside_rect = cv2.bitwise_and(edges, edges, mask=mask)

    # Apply Hough Line Transform
    lines = cv2.HoughLinesP(edges_inside_rect, 1, np.pi / 180, threshold=50, minLineLength=5, maxLineGap=2)

    line_lengths = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        line_lengths.append(length)

    return line_lengths


# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection
edges = cv2.Canny(gray, threshold1=50, threshold2=150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rectangles = []

for contour in contours:
    # Approximate the contour as a polygon
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Check if the polygon has 4 vertices (rectangle)
    if len(approx) == 4:
        rectangles.append(approx)


# Calculate the maximum length among all rectangles
max_length = max([max(measure_inner_line_lengths(rect, image)) for rect in rectangles])

numbered_rectangles = []

for rect in rectangles:
    lengths = measure_inner_line_lengths(rect, image)
    length_sum = sum(lengths)
    # Calculate a weighted number based on the sum of lengths
    number = int(length_sum / max_length * 10)
    numbered_rectangles.append((rect, number))

numbered_rectangles.sort(key=lambda x: x[1])

output_image = image.copy()

for i, (rect, number) in enumerate(numbered_rectangles, start=1):
    cv2.drawContours(output_image, [rect], -1, (0, 255, 0), 2)
    x, y = rect[0][0]  # Access the coordinates of the first point
    cv2.putText(output_image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

cv2.imshow('Numbered Rectangles', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()