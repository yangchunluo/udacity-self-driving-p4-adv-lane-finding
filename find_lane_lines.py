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
    :param img: image pixels array
    :return: a tuple of width and height
    """
    return img.shape[1], img.shape[0]


def output_img(img, path):
    """
    Write image as an output file.
    :param img: image pixels array, in BGR color space or gray scale
    :param path: output file path
    """
    # Recursively creating the directories leading to this path
    dirs = [path]
    for _ in range(2):
        dirs.append(os.path.dirname(dirs[-1]))
    for d in dirs[:0:-1]:
        if not os.path.exists(d):
            os.mkdir(d)
    # If color image, convert to BGR to write (cv2.imwrite takes BGR image).
    # Otherwise it is gray scale.
    if len(img.shape) == 3:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def undistort(img, mtx, dist):
    """
    Correct camera distortion
    :param img: image pixels array, in BGR color space
    """
    return cv2.undistort(img, mtx, dist, None, mtx)


def mask_lane_pixels(img, sobelx_thresh, color_thresh):
    """
    Mask lane pixels using gradient and color space information
    :param img: image pixels array, in BGR color space
    """
    # Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
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
    masked_color = np.dstack((np.zeros_like(sx_binary),  # Red
                              sx_binary,                 # Green
                              s_binary)                  # Blue
                             ) * 255

    # Combining gradient and color thresholding
    masked = np.zeros_like(s_binary)
    masked[(s_binary == 1) | (sx_binary == 1)] = 255

    return masked, masked_color


def transform_perspective(img, src_pts, dst_pts, img_size):
    """
    Perspective transform
    :param img: image pixels array, in BGR color space
    """
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    warped = cv2.warpPerspective(img, M, img_size)
    return warped, M, Minv


def preprocess_image(img, mtx, dist, output_dir, img_fname):
    """
    Preprocessing pipeline for an image.
    :param img: image pixels array, in BGR color space
    :param mtx: for camera calibration
    :param dist: for camera calibration
    :param output_dir: output directory
    :param img_fname: output filename for this image, None for disabling output
    :return: preprocessed image
    """
    img_size = get_img_size(img)  # (width, height)

    # Un-distort image
    undist = undistort(img, mtx, dist)
    if img_fname is not None:
        output_img(undist, os.path.join(output_dir, 'test-undistort', img_fname))

    # Mask lane pixels
    masked, masked_color = mask_lane_pixels(img, sobelx_thresh=(20, 100), color_thresh=(170, 255))
    if img_fname is not None:
        output_img(masked_color, os.path.join(output_dir, 'test-masked_color', img_fname))
        output_img(masked, os.path.join(output_dir, 'test-masked', img_fname))

    # Perspective transform
    # Source points are measured manually from test_images/straight_lines1.jpg by finding a trapezoid.
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
    if img_fname is not None:
        output_img(warped, os.path.join(output_dir, 'test-masked-warped', img_fname))

    # Overlay some intermediate output on the original image for visualization.
    insert_image(img, cv2.resize(masked_color, (img_size[0] // 5, img_size[1] // 5)),
                 x=img_size[0] * .75, y=img_size[1] * 0.1)
    insert_image(img, cv2.resize(warped, (img_size[0] // 5, img_size[1] // 5)),
                 x=img_size[0] * .75, y=img_size[1] * .35)

    return img


def insert_image(canvas, insert, x, y):
    """
    Overlay a small image on a background image as an insert.
    :param canvas: background image
    :param insert: inserted image
    :param x: ROI x position
    :param y: ROI y position
    """
    x = int(x)
    y = int(y)
    if len(insert.shape) == 3:
        canvas[y:y + insert.shape[0], x:x + insert.shape[1]] = insert
    else:
        for i in range(3):
            canvas[y:y + insert.shape[0], x:x + insert.shape[1], i] = insert


def fname_generator(max_num_frame=None):
    """Generator for output filename for each frame"""
    idx = 0
    while True:
        idx += 1
        if max_num_frame and idx > max_num_frame:
            yield None  # Stop producing per-frame image output.
        else:
            yield 'video-frame-{}.jpg'.format(idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibration-file', type=str, required=False, default='./calibration-params.p',
                        help='File path for camera calibration parameters')
    parser.add_argument('--image-dir', type=str, required=False, #default='./test_images',
                        help='Directory of images to process')
    parser.add_argument('--video-file', type=str, required=False, default='project_video.mp4',
                        help="Video file to process")
    x = parser.parse_args()

    # Load camera calibration parameters.
    dist_pickle = pickle.load(open(x.calibration_file, "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    if x.image_dir:
        images = glob.glob(os.path.join(x.image_dir, "*.jpg"))
        for fname in images:
            # Read in image file
            img = cv2.imread(fname)  # BGR
            out = preprocess_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                   mtx, dist, 'output_images', os.path.basename(fname))
            output_img(out, os.path.join('output_images', os.path.basename(fname)))
    elif x.video_file:
        gen = fname_generator(max_num_frame=10)
        clip = VideoFileClip(x.video_file) #.subclip(0, 6)
        write_clip = clip.fl_image(lambda frame:  # RGB
            preprocess_image(frame,  mtx, dist, 'output_images_' + x.video_file, next(gen)))
        write_clip.write_videofile('./project_video_overlay.mp4', audio=False)
