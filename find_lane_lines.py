import argparse
import cv2
import os
import pickle
import glob
import numpy as np
from moviepy.editor import VideoFileClip
from collections import namedtuple


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
        if d and not os.path.exists(d):
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
    Pre-processing pipeline for an image.
    :param img: image pixels array, in BGR color space
    :param mtx: for camera calibration
    :param dist: for camera calibration
    :param output_dir: output directory
    :param img_fname: output filename for this image, None for disabling output
    :return: undistorted image, masked image in color, warped image, Minv (for warping back)
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
        [208.0, 720.0],  # Bottom left
        [1095.0, 720.0]  # Bottom right
    ])
    # Destination points are the corresponding rectangle
    dst_points = np.float32([
        [260, 0],
        [980, 0],
        [260, img_size[1]],
        [980, img_size[1]]
    ])
    warped, _, Minv = transform_perspective(masked, src_points, dst_points, img_size)
    if img_fname is not None:
        output_img(warped, os.path.join(output_dir, 'test-masked-warped', img_fname))

    return undist, masked_color, warped, Minv


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
    if len(insert.shape) < 3:
        insert = cv2.cvtColor(insert.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    canvas[y:y + insert.shape[0], x:x + insert.shape[1]] = insert


def fname_generator(max_num_frame=None):
    """Generator for output filename for each frame"""
    idx = 0
    while True:
        idx += 1
        if max_num_frame and idx > max_num_frame:
            yield None  # Stop producing per-frame image output.
        else:
            yield 'video-frame-{}.jpg'.format(idx)


def detect_lane(warped, num_windows, margin, recenter_threshold):
    """
    Detect lane pixels
    :param warped: binary warped image
    :param num_windows: number of sliding windows on Y
    :param margin: window margin on X
    :param recenter_threshold: a tuple of (t1, t2), if # pixels in the window < t1, recenter window back to base
                               if # pixels in the window > t2, recenter window to the mean the current window
    :return:
    """
    ym_per_pixel = 30.0 / 720
    xm_per_pixel = 3.7 / 831

    debug = False
    # Take a histogram of the bottom half of the image
    histogram = np.sum(warped[warped.shape[0] // 2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Create a color image to draw on and visualize the result
    canvas = cv2.cvtColor(warped.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Create a color image to draw the lane region
    region = cv2.cvtColor(np.zeros_like(warped).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Create a color image to draw the lane pixels
    pixels = cv2.cvtColor(np.zeros_like(warped).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    class LaneInfo(namedtuple("LaneInfo",
                              ["x_current", "nonzero_idxs", "fit_coeff"])):
        """Represent left or right lane information"""

    left_lane = LaneInfo([leftx_base], [], [])
    right_lane = LaneInfo([rightx_base], [], [])

    # Set height of windows
    window_height = np.int(warped.shape[0] / num_windows)

    # Slide through the windows one by one
    for w in range(num_windows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped.shape[0] - (w + 1) * window_height
        win_y_high = warped.shape[0] - w * window_height

        # For left and right lanes
        for lane, which in [(left_lane, "left"), (right_lane, "right")]:
            win_x_low = lane.x_current[0] - margin
            win_x_high = lane.x_current[0] + margin

            # Draw the window for visualization
            cv2.rectangle(canvas, (win_x_low, win_y_low), (win_x_high, win_y_high),
                          (0, 255, 0), thickness=2)

            # Identify the nonzero pixels in x and y within the window
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
            # Append these indices to the lists
            lane.nonzero_idxs.append(good_inds)

            # If number of good pixels > the threshold, recenter next window on their mean position.
            if len(good_inds) > recenter_threshold[1]:
                lane.x_current[0] = np.int(np.mean(nonzerox[good_inds]))
                debug and print(w, which, 'updated', lane.x_current)
            # If number of good pixels < the threshold, recenter next window to base.
            elif len(good_inds) < recenter_threshold[0]:
                lane.x_current[0] = leftx_base if which == "left" else rightx_base
                debug and print(w, which, "reverted", lane.x_current)
            else:
                debug and print(w, which, 'remained', lane.x_current)

    def poly_value(yval, coeffs):
        return coeffs[0] * yval ** 2 + coeffs[1] * yval + coeffs[2]

    def radius_of_curvature(yval, coeffs):
        return ((1 + (2 * coeffs[0] * yval + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])

    # Fit the curve
    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    region_pts = []
    lane_radius = []
    lane_xpos = []
    for lane, color, which in [(left_lane, [255, 0, 0], 'left'),
                               (right_lane, [0, 0, 255], 'right')]:
        # Concatenate the arrays of indices
        nonzero_idx = np.concatenate(lane.nonzero_idxs)
        # Extract line pixel positions
        lane_x = nonzerox[nonzero_idx]
        lane_y = nonzeroy[nonzero_idx]
        # Color the pixels for visualization
        canvas[lane_y, lane_x] = color
        pixels[lane_y, lane_x] = color

        # Fit a second order polynomial on pixel distance
        coeff_pixel = np.polyfit(lane_y, lane_x, deg=2)

        # Fit a second order polynomial on world distance
        coeff_world = np.polyfit(lane_y * ym_per_pixel, lane_x * xm_per_pixel, deg=2)

        # Visualize on the fitted curve on the canvas
        fitx = poly_value(ploty, coeff_pixel)
        # The equivalent of matplotlib.pyplot.plot(X, Y)
        for x, y in zip(fitx, ploty):
            cv2.circle(canvas, center=(int(x), int(y)), radius=3, color=[255, 255, 0], thickness=2)

        # Generate the polygon points to draw the fitted lane region.
        pts = np.transpose(np.vstack([fitx, ploty]))
        if which == 'right':
            # So that when h-stacked later, the bottom left lane is adjacent to the bottom right lane (U-shape).
            pts = np.flipud(pts)
        region_pts.append(np.array([pts]))  # Don't miss the [] around pts

        # Compute radius of curvature and lane X position where the vehicle is (bottom of the view).
        curv = radius_of_curvature(np.max(ploty) * ym_per_pixel, coeff_world)
        debug and print(which, "curvature", curv)
        lane_radius.append(curv)
        lane_xpos.append(poly_value(np.max(ploty), coeff_pixel))

    # Draw the region between left and right lanes.
    cv2.fillPoly(region, np.int_([np.hstack(region_pts)]), (0, 255, 0))

    avg_radius = np.mean(lane_radius)
    dist_center = (midpoint - np.mean(lane_xpos)) * xm_per_pixel
    return canvas, region, pixels, avg_radius, dist_center


def process_pipeline(img, mtx, dist, output_dir, img_base_fname):
    img_size = get_img_size(img)

    # Preprocess image
    undist, masked_color, warped, Minv = preprocess_image(img, mtx, dist, output_dir, img_base_fname)

    # Detect lane
    window, region, pixels, radius, distance = detect_lane(
        warped, num_windows=9, margin=75, recenter_threshold=(10, 100))

    # Overlay intermediate outputs on the original image.
    insert_image(undist, cv2.resize(masked_color, (img_size[0] // 3, img_size[1] // 3)),
                 x=img_size[0] * .65, y=img_size[1] * .03)
    insert_image(undist, cv2.resize(window, (img_size[0] // 3, img_size[1] // 3)),
                 x=img_size[0] * .65, y=img_size[1] * .40)

    # Highlight the lanes on the original image. First warp it back.
    warpback = cv2.warpPerspective(region, Minv, img_size)
    result = cv2.addWeighted(undist, 1, warpback, 0.3, 0)
    warpback = cv2.warpPerspective(pixels, Minv, img_size)
    result = cv2.addWeighted(result, .8, warpback, 1, 0)

    # Add text for curvature and distance to lane center
    cv2.putText(result, "Radius of Curvature: {:.1f}(m)".format(radius), org=(80, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=3, color=(255, 255, 255))
    cv2.putText(result, "Distance to Center: {:.3f}(m) {}".format(
        abs(distance), 'left' if distance < 0 else 'right'), org=(80, 100),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=3, color=(255, 255, 255))

    return result


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
        #images = ['./test_images/test4.jpg']
        for fname in sorted(images):
            print(fname)
            img = cv2.imread(fname)  # BGR
            out = process_pipeline(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mtx, dist,
                                   'output_images', os.path.basename(fname))
            output_img(out, os.path.join('output_images', os.path.basename(fname)))

    if x.video_file:
        gen = fname_generator(max_num_frame=10)
        clip = VideoFileClip(x.video_file)  #.subclip(0, 2)
        write_clip = clip.fl_image(lambda frame:  # RGB
            process_pipeline(frame,  mtx, dist, 'output_images_' + x.video_file, next(gen)))
        write_clip.write_videofile('./project_video_overlay.mp4', audio=False)
