# Plate_detection

1).Import Required Libraries:
        from matplotlib import pyplot as plt
        import imutils
        import cv2
        import easyocr
        import numpy as np

This section imports the necessary libraries for image processing, visualization, and OCR.

2).Load and Preprocess the Image:
        img = cv2.imread('image1.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
The script loads an image named 'image1.jpg' and converts it to grayscale for further processing.

Noise Reduction and Edge Detection:
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Apply bilateral filter for noise reduction
        edged = cv2.Canny(bfilter, 30, 200)  # Perform edge detection
The script applies bilateral filtering to reduce noise and then uses the Canny edge detection algorithm to detect edges in the image.

Find and Sort Contours:

      keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      contours = imutils.grab_contours(keypoints)
      contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
The code finds contours in the edge-detected image and sorts them based on their areas. It keeps the largest 10 contours.

Find License Plate Location:
location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break
The script iterates through the sorted contours and approximates each contour with a polygon. It looks for a polygon with 4 vertices, which is likely the license plate.


Create a Mask and Extract Cropped Image:
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0, 255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)
(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2 + 1, y1:y2 + 1]
A mask is created to highlight the license plate area. The script then uses the mask to extract the license plate region from the original image.


Perform OCR on the Cropped License Plate:
reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)
text = result[0][-2]
The EasyOCR library is used to perform optical character recognition (OCR) on the cropped license plate image. The recognized text is extracted from the OCR result.


Annotate and Display the Result:
font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1] + 60), fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)
The recognized text is added to the original image using OpenCV's putText function, and a rectangle is drawn around the detected license plate region.


Write OCR Result to a File:
file=open('requirements.txt', 'w')
file.write(text)
file.close()
The recognized license plate number is written to a file named 'requirements.txt'.
