import cv2

img = cv2.imread("test.png")  # 아무 이미지
cv2.imshow("img", img)
cv2.waitKey(1)
roi = cv2.selectROI("img", img, False, False)
print("Selected ROI:", roi)
cv2.destroyAllWindows()
