# Automatic License Plate Recognition (ANPR) Using OpenCV and Tesseract OCR

This project demonstrates an Automatic Number Plate Recognition (ANPR) system implemented in Python using OpenCV for image processing and Tesseract OCR for character recognition.

## Features

- Detects license plates in images using contour detection and morphological image processing.
- Extracts and preprocesses the license plate region to improve OCR accuracy.
- Performs OCR with Tesseract to read license plate characters.
- Includes heuristics to handle common OCR misclassifications.
- Logs recognized license plates with owner information and timestamp.
- Visual debugging windows to inspect intermediate processing steps.

## Requirements

- Python 3.7+
- OpenCV (`opencv-python`)
- NumPy
- imutils
- pytesseract
- Tesseract OCR executable installed and accessible in your system path

## Installation

1. Install Python packages:
    pip install opencv-python numpy imutils pytesseract
2. Install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) on your machine.
3. Update the Tesseract executable path in the script if required (default in Windows):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

## Usage

- Place car images in the project directory (default example file: `person2car.jpg`).
- Run the script.
- The program opens windows displaying intermediate steps—press any key to advance.
- Recognized license plates and their owners (from a predefined dictionary) are printed and logged to `NUMPLATE DATA.txt`.

## How It Works

1. The input image is converted to grayscale.
2. Morphological operations highlight regions with intensity differences typical for plates.
3. Sobel gradients and thresholding isolate potential plate regions.
4. Contours are found and filtered by size, shape (quadrilateral), and aspect ratio.
5. The largest suitable contour is extracted as the plate region, which is further cleaned and thresholded.
6. The region is denoised and sharpened before OCR.
7. OCR with multiple Tesseract Page Segmentation Modes attempts to read the plate text.
8. Common OCR confusions like ‘O’/’0’ and ‘S’/’5’ are corrected heuristically.
9. The text is matched against a predefined dictionary of license plates for owner identification.
10. Output is shown and logged with timestamps.

## Project Structure

- `ANPR_final.py`: Main script with the license plate detection and recognition logic.
- `person2car.jpg`: Sample input image (place your test images or update the path).
- `NUMPLATE DATA.txt`: Log file updating recognized plates with timestamps.
- `debug_plate_for_ocr.jpg`: Saved debug image of preprocessed plate for OCR (generated per run).

## Known Limitations

- Works best on reasonably clear, frontal images of license plates with minimal occlusions.
- OCR accuracy depends heavily on image quality and preprocessing.
- Limited to predefined license plates in the `data` dictionary.
- May struggle with complex backgrounds or unusual plate styles.
- For robust real-time use, advanced deep learning models are recommended.