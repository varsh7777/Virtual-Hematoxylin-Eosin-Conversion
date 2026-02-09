import argparse
from pathlib import Path
from typing import List, Union
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# constants/goals
## HUE 
h_cytoplasm_lower = 210
h_nuclei_lower = 172
h_white_lower = 90

h_cytoplasm_upper = 6
h_nuclei_upper = 255
h_white_upper = 152

## SATURATION 
s_cytoplasm_lower = 13
s_nuclei_lower = 13
s_white_lower = 0

s_cytoplasm_upper = 120
s_nuclei_upper = 120
s_white_upper = 44

## BRIGHTNESS
b_cytoplasm_lower = 83
b_nuclei_lower = 38
b_white_lower = 172

b_cytoplasm_upper = 166
b_nuclei_upper = 160
b_white_upper = 207

def rgb_hist(path):
    # returns an rgb histogram

    # load image
    imageObj = cv2.imread(path)

    blue_color = cv2.calcHist([imageObj], [0], None, [256], [0, 256])
    red_color = cv2.calcHist([imageObj], [1], None, [256], [0, 256])
    green_color = cv2.calcHist([imageObj], [2], None, [256], [0, 256])

    cv2.normalize(blue_color, blue_color, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(green_color, green_color, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(red_color, red_color, 0, 1, cv2.NORM_MINMAX)

    # combined histogram
    plt.title("rgb histogram")
    plt.xlabel('pixels')
    plt.ylabel('freq')
    plt.plot(blue_color, color="blue")
    plt.plot(green_color, color="green")
    plt.plot(red_color, color="red")
    plt.show()

def hue_hist(path):
    # returns an hue histogram
    imageObj = cv2.imread(path)
    hsv = cv2.cvtColor(imageObj, cv2.COLOR_BGR2HSV)
    hue_histogram = cv2.calcHist([hsv], [0], None, [180], [0, 180])

    cv2.normalize(hue_histogram, hue_histogram, 0, 1, cv2.NORM_MINMAX)

    plt.title("hue histogram")
    plt.xlabel('pixels')
    plt.ylabel('freq')
    plt.plot(hue_histogram, color="skyblue")
    plt.show()
    return

def saturation_hist(path):
    # returns a saturation histogram
    imageObj = cv2.imread(path)
    hsv = cv2.cvtColor(imageObj, cv2.COLOR_BGR2HSV)
    s_histogram = cv2.calcHist([hsv], [1], None, [180], [0, 180])

    cv2.normalize(s_histogram, s_histogram, 0, 1, cv2.NORM_MINMAX)

    plt.title("saturation histogram")
    plt.xlabel('pixels')
    plt.ylabel('freq')
    plt.plot(s_histogram, color="maroon")
    plt.show()
    return

def brightness_hist(path):
    # returns a brightness histogram
    imageObj = cv2.imread(path)
    hsv = cv2.cvtColor(imageObj, cv2.COLOR_BGR2HSV)
    b_histogram = cv2.calcHist([hsv], [2], None, [180], [0, 180])

    cv2.normalize(b_histogram, b_histogram, 0, 1, cv2.NORM_MINMAX)

    plt.title("brightness histogram")
    plt.xlabel('pixels')
    plt.ylabel('freq')
    plt.plot(b_histogram, color="orange")
    plt.show()
    return