import os
import sys
import cv2
import numpy as np

from rich import print
from scipy import optimize


def putBText(img,text, position=(20, 20),vspace=10,hspace=10, font_scale=1.0,background_RGB=(228,225,222),text_RGB=(1,1,1),font = cv2.FONT_HERSHEY_SIMPLEX,thickness = 2,alpha=0.6,gamma=0):
    """
    source: https://pypi.org/project/pyshine/
    Inputs:
    img: cv2 image img
    text_offset_x, text_offset_x: X,Y location of text start
    vspace, hspace: Vertical and Horizontal space between text and box boundries
    font_scale: Font size
    background_RGB: Background R,G,B color
    text_RGB: Text R,G,B color
    font: Font Style e.g. cv2.FONT_HERSHEY_DUPLEX,cv2.FONT_HERSHEY_SIMPLEX,cv2.FONT_HERSHEY_PLAIN,cv2.FONT_HERSHEY_COMPLEX
            cv2.FONT_HERSHEY_TRIPLEX, etc
    thickness: Thickness of the text font
    alpha: Opacity 0~1 of the box around text
    gamma: 0 by default

    Output:
    img: CV2 image with text and background
    """
    img_height, img_width, _ = img.shape
    R,G,B = background_RGB[0],background_RGB[1],background_RGB[2]
    text_R,text_G,text_B = text_RGB[0],text_RGB[1],text_RGB[2]
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)[0]
    x, y, w, h = position[0], position[1], text_width , text_height

    img_y_min = np.maximum(0,y-vspace)
    img_y_max = np.minimum(y+h+vspace, img_height)

    img_x_min = np.maximum(0, x-hspace)
    img_x_max = np.minimum(x+w+hspace, img_width)

    # crop = img[y-vspace:y+h+vspace, x-hspace:x+w+hspace]
    crop = img[img_y_min:img_y_max, img_x_min:img_x_max]
    white_rect = np.ones(crop.shape, dtype=np.uint8)
    b,g,r = cv2.split(white_rect)
    rect_changed = cv2.merge((B*b,G*g,R*r))


    res = cv2.addWeighted(crop, alpha, rect_changed, 1-alpha, gamma)
    img[img_y_min:img_y_max, img_x_min:img_x_max] = res

    cv2.putText(img, text, (x, (y+h)), font, fontScale=font_scale, color=(text_B,text_G,text_R ), thickness=thickness)
    return img
