import argparse
import cv2
import os
import pickle
import glob
import numpy as np
from moviepy.editor import VideoFileClip


def get_img_size(img):
    """
    Get image size
    :param img: OpenCV image
    :return: a tuple of width and height
    """
    return img.shape[1], img.shape[0]


def output_img(img, dir, filename):
    """
    Write image as an output file.
    :param img: OpenCV image
    :param dir: output directory, create if not exists
    :param filename: filename
    """
    if not os.path.exists(dir):
        print("Creating " + dir)
        os.mkdir(dir)
    out_fname = os.path.join(dir, filename)
    print("Writing " + out_fname)
    cv2.imwrite(out_fname, img)


def undistort(img, mtx, dist):
    """Correct camera distortion"""
    return cv2.undistort(img, mtx, dist, None, mtx)


def mask_lane_pixels(img, sobelx_thresh, color_thresh):
    """Mask lane pixels using gradient and color space information"""
    img = np.copy(img)

    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_GRB2HLS).astype(np.float)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Sobel X on L-channel
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold on X gradient
    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel >= sobelx_thresh[0]) & (scaled_sobel <= sobelx_thresh[1])] = 1

    # Threshold on S-channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= color_thresh[0]) & (s_channel <= color_thresh[1])] = 1

    # Stack each channel for visualization
    masked_color = np.dstack((np.zeros_like(sx_binary),  # Blue
                              sx_binary,                 # Green
                              s_binary)                  # Red
                             ) * 255                     # cv2.imwrite is BGR

    # Combining gradient and color thresholding
    masked = np.zeros_like(s_binary)
    masked[(s_binary == 1) | (sx_binary == 1)] = 255

    return masked, masked_color


def transform_perspective(img, src_pts, dst_pts, img_size):
    """Perspective transform"""
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    warped = cv2.warpPerspective(img, M, img_size)
    return warped, M, Minv


def preprocess_image(img, mtx, dist, output_dir, output_fname):
    """
    Preprocessing pipeline for an image.
    :param img: OpenCV read-in image, in BGR color space
    :param mtx: for camera calibration
    :param dist: for camera calibration
    :param output_dir: output directory
    :param output_fname: output filename, None for disabling output
    :return: preprocessed image
    """
    img_size = get_img_size(img)  # (width, height)

    # Un-distort image
    undist = undistort(img, mtx, dist)
    if output_fname is not None:
        output_img(undist, os.path.join(output_dir, 'test-undistort'), output_fname)

    # Mask lane pixels
    masked, masked_color = mask_lane_pixels(img, sobelx_thresh=(20, 100), color_thresh=(170, 255))
    if output_fname is not None:
        output_img(masked_color, os.path.join(output_dir, 'test-masked_color'), output_fname)
        output_img(masked, os.path.join(output_dir, 'test-masked'), output_fname)

    # Perspective transform
    # Source points are measured manually from test_images/straight_lines1.jpg by finding a trapezoid
    src_points = np.float32([
        [576.0, 463.5],  # Top left
        [706.5, 463.5],  # Top right
        [232.0, 700.0],  # Bottom left
        [1069.0, 700.0]  # Bottom right
    ])
    # Destination points are the corresponding rectangle
    dst_points = np.float32([
        [260, 0],
        [980, 0],
        [260, img_size[1]],
        [980, img_size[1]]
    ])
    warped, M, Minv = transform_perspective(masked, src_points, dst_points, img_size)
    if output_fname is not None:
        output_img(warped, os.path.join(output_dir, 'test-masked-warped'), output_fname)

    return undist

idx = 0
def get_fname():
    global idx



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibration-file', type=str, required=False, default='./calibration-params.p',
                        help='File path for camera calibration parameters')
    parser.add_argument('--image-dir', type=str, required=False,
                        help='Directory of images to process')
    parser.add_argument('--video-file', type=str, required=False, default='./project_video.mp4',
                        help="Video file to process")
    x = parser.parse_args()

    dist_pickle = pickle.load(open(x.calibration_file, "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    if x.image_dir:
        images = glob.glob(os.path.join(x.image_dir, "*.jpg"))
        for fname in images:
            # Read in image file
            img = cv2.imread(fname)  # BGR
            preprocess_image(img, mtx, dist, os.path.basename(fname))
    elif x.video_file:
        clip = VideoFileClip(x.video_file).subclip(0, 1)
        write_clip = clip.fl_image(lambda frame:
                                   preprocess_image(frame,  # RGB
                                                    mtx, dist, None))
        write_clip.write_videofile('./test-video.mp4', audio=False)
