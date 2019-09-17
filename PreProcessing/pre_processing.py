import cv2
import os
import numpy as np
#import matplotlib.pylot as plt



def show_image(img, name='Label'):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def edge_detection(img):
    img = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)
    img = cv2.Canny(img, 100, 100)
    return img

def dct_transform(img, blur=False):
    if blur:
        img = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.float32(img) / 255.0
    img = cv2.dct(img)
    return(img)

def high_band_pass(img):
    int_img = np.int32(np.float32(dct_image) * 255.0 )
    max_val = np.max(int_img)
    min_val = np.min(int_img)
    avg_val = np.average(int_img)
    img = img[img < avg_val]  = 0
    return img



if __name__ == '__main__':
    DATA_DIR = "C:\\Users\\Austin\\Pictures\\jaffe\jaffe\\"

    image_name = 'YM.SU3.60.tiff'
    image_path = DATA_DIR + image_name

    image = cv2.imread(image_path)

    init_image = image

    edge_image = edge_detection(init_image)
    show_image(edge_image, 'Edge Image')

    dct_image = dct_transform(init_image, blur=True)
    show_image(dct_image, 'DCT Image')

    filtered_image = high_band_pass(dct_image)

    returned_image = cv2.dct(filtered_image, cv2.DCT_INVERSE)
    show_image(returned_image, 'returned image')

