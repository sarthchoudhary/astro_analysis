# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 14:44:18 2018

@author: Sarth.choudhary
"""

#Limb darkening correction
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.optimize import leastsq
import cv2

plt.close('all')

image_path = r'D:\IIA\Internship\Kodaikanal\H alpha20170803\Ha_20170803_103334940.fits'
flat_path = r'D:\IIA\Internship\Kodaikanal\H alpha20170803\Flat_20170803_112801100.fits'
dark_path = r'D:\IIA\Internship\Kodaikanal\H alpha20170803\Dark_20170803_133407830.fits'

def fitsread(file_path):
    return fits.getdata(file_path)
test_image = fitsread(image_path)
test_image = np.squeeze(test_image)
#plt.figure()
#plt.imshow(test_image, cmap = 'gray')

flat = fitsread(flat_path)
dark = fitsread(dark_path)
def flat_fielding(test_image, dark, flat):
    return (test_image-dark)/(flat-dark)
flat_image = flat_fielding(test_image,dark,flat)
flat_image= np.squeeze(flat_image)
plt.figure()
plt.imshow(flat_image, cmap='gray')
plt.title('Flat field corrected image')

intensity = []
for i in range(test_image.shape[1]):
    intensity.append(test_image[int(flat_image.shape[0]/2)][i])
plt.figure()
plt.plot(intensity)

#from scipy.ndimage.filters import gaussian_filter
#blurred_image = gaussian_filter(test_image, sigma=2, truncate=0.75)
#plt.figure()
#plt.imshow(blurred_image, cmap='gray')
#plt.title('Gaussian Blurred Image')

def get_rms_image(image):
    kernel_size = 5
    image2 = np.power(image, 2)
    kernel = (np.ones(shape=(kernel_size,kernel_size)))/float(kernel_size)
    image3 = convolve2d(image2, kernel, mode='same')
    return np.sqrt(image3)
rms_image = get_rms_image(flat_image)

def estimate_center(image):
#    rms_image = np.copy(image)
    mean= np.mean(image)
    centroid_col = 0
    centroid_row = 0
    weightage = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j] >= mean:
                centroid_col = i*image[i][j] + centroid_col
                centroid_row = j*image[i][j] + centroid_row
                weightage = image[i][j] + weightage
    centroid_row = centroid_row/weightage
    centroid_col = centroid_col/weightage
    return centroid_row, centroid_col
guess_center = estimate_center(rms_image)

'''sobel filter operation'''

def detect_edge(img):
    sobelx = ([[-1,0,1], [-2,0,2], [-1,0,1]])
    sobely = ([[-1,-2,-1],[0,0,0],[1,2,1]])
    gradimgx = np.copy(img)
    gradimgy = np.copy(img)
    gradimgx = convolve2d(gradimgx, sobelx, mode='same')/4.0
    gradimgy = convolve2d(gradimgy, sobely, mode='same')/4.0
    anglegrad = np.arctan2(gradimgy, gradimgx)
    maggrad = np.sqrt(np.square(gradimgy) + np.square(gradimgx))
    threshhold = np.histogram(maggrad)[1][-2]
    isEdge = np.greater_equal(maggrad, threshhold)
    return isEdge, anglegrad, maggrad

isEdge,anglegrad,maggrad = detect_edge(rms_image)
plt.figure()
plt.imshow(maggrad, cmap='gray')
plt.title('Magnitude of Gradient_Image')
#plt.figure()
#plt.imshow(isEdge, cmap='gray')

maggrad_hist = np.histogram(maggrad)            
boundary_points = np.where(maggrad >= (maggrad_hist[1][-2]))
size = boundary_points[0].size
x_coor =[]
y_coor = []
for i in range(size):
    x_coor.append(boundary_points[0][i])
    y_coor.append(boundary_points[1][i])
    
def calc_R(xc,yc):
    return np.sqrt((xc-x_coor)**2 + (yc-y_coor)**2)
def f_2(c):
    Ri = calc_R(*c)
    return Ri -Ri.mean()

center_2, ier = leastsq(f_2, guess_center)
xc_2, yc_2 = center_2                                         #accurate center
Ri_2       = calc_R(*center_2)
R_2        = Ri_2.mean()                                      #mean radius
residu_2   = sum((Ri_2 - R_2)**2)

center_row=np.int(center_2[0])
center_col=np.int(center_2[1])
int_radius = int(R_2)

edge_image = np.copy(flat_image)
cv2.circle(edge_image,(center_row,center_col), np.int(R_2), (1,1,1), 1,8,0)
plt.figure()
plt.imshow(test_image, cmap='gray')
plt.title('detected edge')

intensity1 =[]
#x_ax = range(int(center_2[1] - R_2), int(center_2[1] + R_2 + 1))
x_ax = range(int(center_2[1]), int(center_2[1] + R_2 + 1))
for i in x_ax:
    intensity1.append(flat_image[int(center_2[0])][i])
plt.figure()
plt.plot(intensity1)
plt.title('intensity along diameter only')

poly_coeff = np.polyfit(x_ax, intensity1, 2)
fitted_poly = np.poly1d(poly_coeff) #generates polynomial
fitted_intensity = fitted_poly(x_ax)
plt.figure()
plt.plot(fitted_intensity)
plt.title('fitted curve')

def limb_correction(image):
    correct_image = np.copy(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            R_dist = np.sqrt((xc_2 - j)**2 + (yc_2 -  i)**2)
            if R_dist <= R_2:
                correct_image[i][j] = (correct_image[i][j])/fitted_intensity[int(np.round(R_dist))]
    return correct_image
limb_corrected = limb_correction(flat_image)
plt.figure()
plt.imshow(limb_corrected, cmap='gray')
plt.title('Limb corrected Image')

intensity2=[]
for i in range(test_image.shape[1]):
    intensity2.append(limb_corrected[int(center_2[0])][i])
plt.figure()
plt.plot(intensity2)
plt.title('Intensity Profile of corrected image')
