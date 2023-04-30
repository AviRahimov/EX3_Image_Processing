# import itertools
# import math
# import sys
from typing import List

import numpy as np
import cv2
# import pygame
# from numpy.linalg import LinAlgError
# import matplotlib.pyplot as plt
import warnings

# from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    """
    return 214423147


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------

def myOpticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10,
                  win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10,
                win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """


def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int,
                     stepSize: int, winSize: int) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def find_vec_of_transformMed(mat: np.ndarray):
    """
    This method calculates the median of the (u,v) vector from @mat.
    :param mat: array of the vectors of the OP
    :return: the median vector - (u,v)
    """


def find_vec_of_transform(img1: np.ndarray, img2: np.ndarray, mat: np.ndarray):
    """
    This function returns best vector that suits the translation between @img1 to @img2.
    :param img1: first image
    :param img2: second image
    :param mat: array of the vectors of the OP
    :return: vector - [u,v]
    """


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """


def bestAngle(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    This function go over all the possibilities for an angle between two images (0-359).
    :param img1: first image
    :param img2: second image
    :return: the best angle
    """


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """


def findCorrelation(img1: np.ndarray, img2: np.ndarray):
    """
    This function looks for two points, one from @img1 and second from @img2.
    The two points are the ones with the highest correlation.
    :param img1: first image
    :param img2: second image
    :return: 2 points - x1, y1, x2, y2
    """


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """


def getAngle(point1, point2):
    """
    This function calculate the angle between @point1 to @point2 by checking the intersection
    point of these points by creating two lines to the mass of center (0,0).
    :param point1: [x,y]
    :param point2: [x,y]
    :return: float - angle
    """


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Resorts the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    """
    Expands a Gaussian pyramid level one step up
    :param img: Pyramid image at a certain level
    :param gs_k: The kernel to use in expanding
    :return: The expanded level
    """


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
