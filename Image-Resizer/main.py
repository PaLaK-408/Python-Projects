import cv2

src = cv2.imread("pic.jpg", cv2.IMREAD_UNCHANGED)
cv2.imshow("title", src)

#percent by which the image is resized
scale_percent = 30

#calculate the 30 percent of original dimensions
width = int(src.shape[1] * scale_percent / 100)
height = int(src.shape[0] * scale_percent / 100)

#dsize
dsize = (width, height)

#resize image
output = cv2.resize(src, dsize=dsize, interpolation=cv2.INTER_AREA)

cv2.imwrite("output.jpg", output)
cv2.waitKey(0)


# for newImage
import cv2

#Configuration Parameters
source = "pic.jpg"
destination = "newImage.png"

src = cv2.imread(source, cv2.IMREAD_UNCHANGED)
#cv2.imshow("title",src)

#percent by which the immage is resized
scale_percent = 30

#calculate the 30 percent of original dimensions
new_width = int(src.shape[1] * scale_percent / 100)
new_height = int(src.shape[0] * scale_percent / 100)

# resize image
output = cv2.resize(src, dsize=dsize, interpolation=cv2.INTER_AREA)

cv2.imwrite(destination, output)
cv2.waitKey(0)