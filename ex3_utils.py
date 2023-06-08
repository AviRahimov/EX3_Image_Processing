# import itertools
# import math
# import sys
from typing import List

import numpy as np
import cv2
# from numpy.linalg import LinAlgError
# import matplotlib.pyplot as plt
import warnings

# from sklearn.metrics import mean_squared_error
import scipy
from scipy.signal import correlate2d

warnings.filterwarnings('ignore')


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    """
    return 214423147


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------

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
    # X and Y kernel gradients to convolve with im2 image
    x_deriv_kernel = np.array([[-1, 0, 1]])
    y_deriv_kernel = x_deriv_kernel.T
    # Calculating the gradients for x and y and the difference between the times of im2 and im1
    Ix = cv2.filter2D(im2, -1, x_deriv_kernel, borderType=cv2.BORDER_REPLICATE)
    Iy = cv2.filter2D(im2, -1, y_deriv_kernel, borderType=cv2.BORDER_REPLICATE)
    It = im2 - im1

    # calculating the half size of the window such that we can run over all the window in the for loop
    half_window_size = win_size // 2
    # Initializing the lists for the good original point and the vector for each point, by saying good point I meant
    # to the points that are following after 2 statements of the eigen-values represented in the PDF.
    original_points = []
    vector_for_points = []
    # Traverse on all im1 with windowed size and calculating A(transpose)A = A(transpose)b
    for i in range(half_window_size, im1.shape[0] - half_window_size + 1, step_size):
        for j in range(half_window_size, im1.shape[1] - half_window_size + 1, step_size):
            start_rows_window = i - half_window_size
            end_rows_window = i + half_window_size + 1
            start_cols_window = j - half_window_size
            end_cols_window = j + half_window_size + 1

            Ix_window = Ix[start_rows_window: end_rows_window, start_cols_window: end_cols_window].flatten()
            Iy_window = Iy[start_rows_window: end_rows_window, start_cols_window: end_cols_window].flatten()
            It_window = It[start_rows_window: end_rows_window, start_cols_window: end_cols_window].flatten()

            A = np.vstack((Ix_window, Iy_window)).T  # A = [Ix, Iy]
            AT_A = A.T @ A

            eigen_values = np.linalg.eigvals(AT_A)
            lamda1 = eigen_values.min()
            lamda2 = eigen_values.max()
            # Checking if the original point is good or not by 2 statements of the eigen-values
            if lamda1 <= 1 or lamda2 / lamda1 >= 100:
                continue

            # If the eigen-values are good so, we can continue to calculate A(transpose) * b
            AT_b = (A.T @ (-1 * It_window).T).reshape(2, 1)
            # As A(transpose) * A is a squared matrix, we can calculate the inverse of it and multiply it from the left
            # ,and then we get u and v i.e. the vectors for the good original points i and j.
            u_v = np.linalg.inv(AT_A) @ AT_b
            vector_for_points.append([u_v[0, 0], u_v[1, 0]])
            original_points.append([j, i])
    return np.array(original_points), np.array(vector_for_points)


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
    # Create pyramids for both img1 and img2 by using gaussianPyr function which is implemented later
    pyr_img1 = gaussianPyr(img1, k)
    pyr_img2 = gaussianPyr(img2, k)

    # Computing optical flow for the first level of the pyramid i.e. the smallest image
    original_points, vector_points = opticalFlow(pyr_img1[-1], pyr_img2[-1], stepSize, winSize)

    # Looping through the levels of the pyramid from second-to-last level to the base level
    for i in range(k - 2, -1, -1):
        # Computing optical flow for each level of the pyramid
        orig_pts, vec_pts = opticalFlow(pyr_img1[i], pyr_img2[i], stepSize, winSize)

        # Warp lower level by 2(u, v)
        original_points *= 2
        vector_points *= 2

        # Looping through the points and vectors of the current level
        for pixel, uv_current in zip(orig_pts, vec_pts):
            # Checking if the pixel exists in the original points
            index = np.where((original_points == pixel).all(axis=1))[0]
            if len(index) == 0:
                # Appending the new pixel and vector to the original points and vectors to complete the warping process
                original_points = np.vstack([original_points, pixel])
                vector_points = np.vstack([vector_points, uv_current])
            else:
                # Adding the current vector to the existing vector for the pixel
                vector_points[index[0]][0] += uv_current[0]
                vector_points[index[0]][1] += uv_current[1]

    # Creating an array for the final result with (m, n, 2) shape
    ans = np.zeros(shape=(img1.shape[0], img1.shape[1], 2))
    # Assigning the vector points to the corresponding positions in the ans array
    ans[original_points[:, 1], original_points[:, 0]] = vector_points

    # Returning the resulting array
    return ans


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------

def findTranslationLK(im1: np.ndarray, im2: np.ndarray, num_iterations=5) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :param num_iterations: number of iterations for iterative refinement.
    :param regularization: regularization parameter to handle singular matrices.
    :return: Translation matrix by LK.
    """
    # Compute image gradients
    Ix = cv2.filter2D(im1, -1, np.array([[-1, 1]]), borderType=cv2.BORDER_REPLICATE)
    Iy = cv2.filter2D(im1, -1, np.array([[-1], [1]]), borderType=cv2.BORDER_REPLICATE)
    It = im2 - im1

    # Initialize translation matrix
    T = np.eye(3)

    for _ in range(num_iterations):
        # Warp image using current translation matrix
        warped_im1 = cv2.warpPerspective(im1, T, (im1.shape[1], im1.shape[0]))

        # Compute error and gradient of warped image
        error = warped_im1 - im2
        Ix_w = cv2.filter2D(warped_im1, -1, np.array([[-1, 1]]), borderType=cv2.BORDER_REPLICATE)
        Iy_w = cv2.filter2D(warped_im1, -1, np.array([[-1], [1]]), borderType=cv2.BORDER_REPLICATE)

        # Compute the elements of the matrix A and vector b with regularization
        A11 = np.sum(Ix_w * Ix_w) + 0.01
        A12 = np.sum(Ix_w * Iy_w)
        A22 = np.sum(Iy_w * Iy_w) + 0.01
        b1 = -np.sum(Ix_w * error)
        b2 = -np.sum(Iy_w * error)

        # Solve the system of equations
        A = np.array([[A11, A12], [A12, A22]])
        b = np.array([b1, b2])
        v = np.linalg.solve(A, b)

        # Limit maximum translation per iteration
        v[0] = np.abs(np.clip(v[0], -10, 10))
        v[1] = np.abs(np.clip(v[1], -10, 10))
        # Update translation matrix
        T[0, 2] -= v[0]
        T[1, 2] -= v[1]

    return T


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    EPS = 0.000001
    min_error = np.inf
    final_rotation = np.eye(3, dtype=np.float32)  # Identity matrix
    directions = opticalFlow(im1, im2)[1]
    final_rotated_img = np.zeros_like(im1)

    for u, v in directions:
        angle = np.arctan2(v, u) * 180 / np.pi  # Convert angle to degrees

        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                    [np.sin(angle), np.cos(angle), 0],
                                    [0, 0, 1]], dtype=np.float32)

        rotated_img = cv2.warpPerspective(im1, rotation_matrix, im1.shape[::-1])  # Rotate the image
        mse = np.mean(np.square(im2 - rotated_img))  # Calculate the mean squared error

        if mse < min_error:
            min_error = mse
            final_rotation = rotation_matrix
            final_rotated_img = rotated_img.copy()

        if mse < EPS:
            break

    translation = findTranslationLK(final_rotated_img, im2)  # Find the translation from the rotated image to im2
    translation[0, 2] = -1  # Set the correct translation value for x-axis
    translation[1, 2] = -1  # Set the correct translation value for y-axis
    final_ans = translation @ final_rotation  # Dot product for getting the rigid matrix
    return final_ans


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    EPS = 0.001
    relevant_gap = 100

    # Generate a grid of translation values
    y_range = np.arange(-relevant_gap, relevant_gap)
    x_range = np.arange(-relevant_gap, relevant_gap)
    translations = np.array(np.meshgrid(y_range, x_range)).T.reshape(-1, 2)

    min_error = np.inf
    final_ans = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]], dtype=np.float32)

    for translation in translations:
        check = np.array([[1, 0, translation[0]],
                          [0, 1, translation[1]],
                          [0, 0, 1]], dtype=np.float32)
        moved_img = cv2.warpPerspective(im1, check, im1.shape[::-1])
        mse = np.square(im2 - moved_img).mean()

        if mse < min_error:
            min_error = mse
            final_ans = check.copy()
        if mse < EPS:
            return final_ans

    return final_ans


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    EPS = 0.000001
    min_error = np.inf
    final_rotation = np.eye(3, dtype=np.float32)  # Initialize as identity matrix
    final_rotated_img = None

    # Iterate over angles in radians instead of degrees
    angles = np.linspace(0, 2 * np.pi, num=360, endpoint=False)
    for alpha in angles:
        rotation_matrix = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                                    [np.sin(alpha), np.cos(alpha), 0],
                                    [0, 0, 1]], dtype=np.float32)
        rotated_img = cv2.warpPerspective(im1, rotation_matrix, im1.shape[::-1])
        mse = np.square(im2 - rotated_img).mean()
        print(mse)
        if mse < min_error:
            min_error = mse
            final_rotation = rotation_matrix.copy()
            final_rotated_img = rotated_img.copy()

        if mse < EPS:
            break

    translation = findTranslationCorr(final_rotated_img, im2)
    final_ans = np.matmul(translation, final_rotation)

    return final_ans


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    height, width = im2.shape

    X, Y = np.meshgrid(range(width), range(height))
    new_img = np.array([X.flatten(), Y.flatten(), np.ones_like(X.flatten())])

    old_img = (np.linalg.inv(T)) @ new_img
    first_mask = (old_img[0, :] > width) | (old_img[0, :] < 0)
    second_mask = (old_img[1, :] > height) | (old_img[1, :] < 0)
    old_img[0, :][first_mask] = 0
    old_img[1, :][second_mask] = 0

    transformed_img = im2[old_img[1, :].astype(int), old_img[0, :].astype(int)]
    transformed_img = transformed_img.reshape((height, width))  # Reshape the transformed image

    return transformed_img



# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    gaussian_pyr = [img]
    for i in range(1, levels):
        I_temp = cv2.GaussianBlur(gaussian_pyr[i - 1], (5, 5), 0)
        I_temp = I_temp[::2, ::2]
        gaussian_pyr.append(I_temp)
    return gaussian_pyr


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    gaussian_pyramid = gaussianPyr(img, levels)
    laplacian_pyr = []
    for i in range(1, levels):
        blur_img_expanded = gaussExpand(gaussian_pyramid[i], (5, 5))
        laplacian_img = cv2.subtract(gaussian_pyramid[i - 1], blur_img_expanded)
        laplacian_pyr.append(laplacian_img)
    return laplacian_pyr


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    orig_img = lap_pyr[-1]
    # reverse traversal i.e. starts with the smallest laplacian image
    for i in range(len(lap_pyr) - 1, 0, -1):
        expanded_orig_img = gaussExpand(orig_img, (5, 5))
        orig_img = cv2.add(lap_pyr[i - 1], expanded_orig_img)
    return orig_img


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray,
             mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
    pass


def gaussExpand(img: np.ndarray, gaussian_k: np.ndarray) -> np.ndarray:
    """
    Performs Gaussian expansion of an input image by a factor of 2 in both dimensions.
    :param img: The input image to be expanded.
    :param gaussian_k: The Gaussian kernel for filtering during expansion.
    :return: The expanded image.
    """

    x, y = img.shape[0], img.shape[1]
    if len(img.shape) == 3:  # RGB
        shape = (x * 2, y * 2, 3)
    else:  # GRAY
        shape = (x * 2, y * 2)
    preImg = np.zeros(shape)
    preImg[::2, ::2] = img
    return cv2.filter2D(preImg, -1, gaussian_k, borderType=cv2.BORDER_REPLICATE)
