""""Parameters
----------
image : ndarray
    Input image data. Will be converted to float.
mode : str
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n,is uniform noise with specified mean & variance.
"""""
import numpy as np
import os
import cv2
import scipy
from PIL import Image, ImageFilter
from scipy import special, optimize, signal
from google.cloud import vision
from google.cloud.vision import types
from google.oauth2 import service_account

noise = True
s_p = True

def noisy(noise_typ,image,amount):

    if noise_typ == "gauss":
        row,col,ch= np.array(image).shape
        mean = 0
        #var = 0.1
        #sigma = var**0.5
        gauss = np.random.normal(mean,1,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = np.array(image).shape
        gray = image.convert('L')
        for i in range(0, row - 1):
            for j in range(0, col - 1):
                rand = np.random.randint(0, 101)
                if rand <= amount:
                    pixel_g = gray.getpixel((i, j))
                    if pixel_g <= 127:
                        image.putpixel((i, j), (255, 255, 255))
                    else:
                        image.putpixel((i, j), (0, 0, 0))
        return image
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = np.array(image).shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)
        noisy = image + image * gauss
        return noisy

def change(img, img_noise):
    size = np.array(img).shape
    row = size[0]
    col = size[1]
    cntr = 0
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            if original.getpixel((i, j)) != img_noise.getpixel((i, j)):
                cntr += 1
    change = cntr / (img.size[0] * img.size[1])
    print("change(%) = ", change)
    return change

def image_iterator():

    credentials = service_account.Credentials.from_service_account_file('service_acc_key.json')
    client = vision.ImageAnnotatorClient(credentials=credentials)

    path = 'E:\wrand\\'
    for image_path in os.listdir(path):

        file_name = path + image_path

        with io.open(file_name, 'rb') as image_file:
            content = image_file.read()

        image = types.Image(content=content)
        if noise:
            if s_p:
                image = noisy('s&p', image)
            else:
                # TBD gauss
                print()


        response = client.label_detection(image=image)
        labels = response.label_annotations
        print('Labels:')
        for label in labels:
            print(label.description)
            print(' ')
            break

def main():

    img = Image.open("cat.jpg")
    original = Image.open("cat.jpg")
    img_noise = noisy("s&p", img, 12)
    img_noise.save("img_noise.jpg")
    #img_noise.show()



    newData = []
    data = img_noise.getdata()
    #print(img_noise.getpixel((i, j))[1])
    median = img_noise.filter(ImageFilter.MedianFilter(3))
    median.show()
    median.save("img_median.jpg")

if __name__ == "__main__": main()