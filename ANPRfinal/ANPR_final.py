from datetime import datetime
import numpy as np
import imutils
import cv2
import pytesseract
import os
import re

data = {'7JCV393': 'PERSON1',
        'KL55R2473': 'PERSON2',
        'HR26DK8337': 'PERSON3'
        }

keep = 10
minAR = 0.8
maxAR = 7.0

script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, "person2car.jpg")
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Car", img)
cv2.imshow("Grayscale", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
cv2.imshow("Blackhat", blackhat)

squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Light Regions", light)

cv2.waitKey(0)
cv2.destroyAllWindows()

gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
gradX = gradX.astype("uint8")
cv2.imshow("Scharr", gradX)

cv2.waitKey(0)
cv2.destroyAllWindows()

gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Grad Thresh", thresh)

thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)
cv2.imshow("Grad Erode/Dilate", thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]

# Draw all contours for debugging
debug_img = img.copy()
for c in cnts:
    cv2.drawContours(debug_img, [c], -1, (0, 255, 0), 2)
cv2.imshow("All candidate contours", debug_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

plateCnt = None
plateArea = 0

for c in cnts:
    area = cv2.contourArea(c)
    if area < 30:
        continue

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if 4 <= len(approx) <= 6:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        print(f"Contour aspect ratio: {ar:.2f}, area: {area}")
        if minAR <= ar <= maxAR:
            if area > plateArea:
                plateArea = area
                plateCnt = approx

if plateCnt is not None:
    x, y, w, h = cv2.boundingRect(plateCnt)
    licensePlate = gray[y:y + h, x:x + w]

    # Inverse threshold for better OCR (black text on white)
    roi = cv2.threshold(licensePlate, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Denoise and sharpen
    denoised = cv2.fastNlMeansDenoising(roi, h=10)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    cv2.imwrite("debug_plate_for_ocr.jpg", sharpened)
    cv2.imshow("License Plate", licensePlate)
    cv2.imshow("Preprocessed for OCR", sharpened)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("License plate not found!")
    exit()

lp_resized = cv2.resize(sharpened, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

def build_tesseract_options(psm=7):
    alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    options = f"-c tessedit_char_whitelist={alphanumeric} --psm {psm}"
    return options

def correct_ocr_errors(text):
    text = text.upper()
    # Common confusions — only apply if exact match not found
    replacements = {
        'Z': '2',
        'S': '5',
        'B': '8',
        'O': '0',
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Keep only A-Z and 0-9
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text

best_lptext = ""
found = False
'''
for psm in [7, 8, 6, 11]:
    options = build_tesseract_options(psm)
    raw_text = pytesseract.image_to_string(lp_resized, config=options)
    clean_text = raw_text.strip("\n\x0c").replace(" ", "").upper()
    print(f"PSM {psm} raw OCR output: '{clean_text}'")

    if clean_text in data:
        best_lptext = clean_text
        found = True
        break
'''
if not found:
    # Try error correction if exact match not found
    for psm in [7, 8, 6, 11]:
        options = build_tesseract_options(psm)
        raw_text = pytesseract.image_to_string(lp_resized, config=options)
        corrected_text = correct_ocr_errors(raw_text)
        ##print(f"PSM {psm} corrected OCR output: '{corrected_text}'")
        if corrected_text in data:
            best_lptext = corrected_text
            found = True
            break
        # Keep longest if no exact match for fallback
        if len(corrected_text) > len(best_lptext):
            best_lptext = corrected_text

owner = data.get(best_lptext, "UNKNOWN")

cv2.imshow("Final ROI", lp_resized)
print("License plate:", best_lptext)
print("Owner:", owner)

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
with open("NUMPLATE DATA.txt", mode="a") as f:
    s = f"{best_lptext} ({owner}) entered at {dt_string}\n"
    print(s)
    f.write(s)

cv2.waitKey(0)
cv2.destroyAllWindows()
