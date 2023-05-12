import cv2 
import numpy as np 
import os

def thresh_crop_image(img) :
    # get the name of the file
    path = os.path.abspath(img)
    name = os.path.basename(img)
    # read the image
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # threshold 
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    hh, ww = thresh.shape
    # make bottom 2 rows black where they are white the full width of the image
    thresh[hh-3:hh, 0:ww] = 0
    # get bounds of white pixels
    white = np.where(thresh==255)
    xmin, ymin, xmax, ymax = np.min(white[1]), np.min(white[0]), np.max(white[1]), np.max(white[0])
    # crop the image at the bounds adding back the two blackened rows at the bottom
    crop = img[ymin:ymax+3, xmin:xmax]
    # save resulting masked image
    cropped_img = cv2.imwrite(path, crop)
    return f"{name} cropped."

def resize_image(img):
    # get the name of the file
    path = os.path.abspath(img)
    name = os.path.basename(img)    
    # read the image
    img = cv2.imread(img)
    # resize and save to new directory
    resize = cv2.resize(img, (500, 500)) 
    resized_image = cv2.imwrite(path, resize)
    
    return f"{name} resized."


def remove_background(img) : 
    # Read the image
    # Import the image
    path = os.path.abspath(img)
    name = os.path.splitext(img)[0]
    # Read the image
    src = cv2.imread(img, 1)
    # Convert image to image gray
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # Applying thresholding technique
    _, alpha = cv2.threshold(tmp, 7, 255, cv2.THRESH_BINARY)
    # Using cv2.split() to split channels 
    # of coloured image
    b, g, r = cv2.split(src)
    
    # Making list of Red, Green, Blue
    # Channels and alpha
    rgba = [b, g, r, alpha]
    
    # Using cv2.merge() to merge rgba
    # into a coloured/multi-channeled image
    dst = cv2.merge(rgba, 4)
    
    # Writing and saving to a new image
    cv2.imwrite(f"{name}.png", dst)
    os.remove(path)

    return f"{name} background is removed."
