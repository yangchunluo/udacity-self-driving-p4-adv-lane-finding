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
        output_img(undist, os.path.join(output_dir, 'undistort', img_fname))

    # Mask lane pixels
    masked, masked_color = mask_lane_pixels(img, sobelx_thresh=(20, 100), color_thresh=(170, 255))
    if img_fname is not None:
        output_img(masked_color, os.path.join(output_dir, 'masked_color', img_fname))
        output_img(masked, os.path.join(output_dir, 'masked', img_fname))

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
        output_img(warped, os.path.join(output_dir, 'masked-warped', img_fname))

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


class LaneInfo:
    def __init__(self):
        # The fit is valid
        self.fit_valid = False
        # Coeffients of the fit
        self.fit_coeff = []


def detect_lane(warped, lane_infos, num_windows, margin_detect, margin_track, recenter_threshold,
                output_dir, img_fname):
    """
    Detect lane pixels
    :param warped: binary warped image
    :param lane_infos: lane information for look-ahead filter, reset, and smoothing, set None for separate images
    :param num_windows: number of sliding windows on Y
    :param margin_detect: margin on X for detecting
    :param margin_track: margin on X for tracking
    :param output_dir: output directory
    :param img_fname: output filename for this image, None for disabling output
    :param recenter_threshold: a tuple of (t1, t2), if # pixels in the window < t1, recenter window back to base
                               if # pixels in the window > t2, recenter window to the mean the current window
    :return:
    """
    debug = False

    # Constant conversion rate from pixel to world distance
    ym_per_pixel = 30.0 / 720
    xm_per_pixel = 3.7 / 831

    # Create a base color image to draw on and visualize the result
    canvas = cv2.cvtColor(warped.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    # Create a color image to draw the lane region
    region = cv2.cvtColor(np.zeros_like(warped).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    # Create a color image to draw the lane pixels
    pixels = cv2.cvtColor(np.zeros_like(warped).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Take a histogram of the bottom half of the image
    histogram = np.sum(warped[warped.shape[0] // 2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)

    xbase = {'left': np.argmax(histogram[:midpoint]),
             'right': np.argmax(histogram[midpoint:]) + midpoint}
    xcurrent = xbase
    color = {'left': [255, 0, 0],
             'right': [0, 0, 255]}
    region_pts = {}
    lane_radius = {}
    lane_xpos = {}

    def poly_value(yval, coeffs):
        return coeffs[0] * yval ** 2 + coeffs[1] * yval + coeffs[2]

    def radius_of_curvature(yval, coeffs):
        return ((1 + (2 * coeffs[0] * yval + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])

    # For left and right lanes
    for which in ["left", "right"]:
        is_tracking = lane_infos is not None and lane_infos[which].fit_valid

        if is_tracking:
            # Tracking mode: search within a margin of previous fitted curve
            nonzero_idx = ((nonzerox > (poly_value(nonzeroy, lane_infos[which].fit_coeff) - margin_track)) &
                            (nonzerox < (poly_value(nonzeroy, lane_infos[which].fit_coeff) + margin_track)))

        else:
            # Detecting mode: sliding window
            nonzero_idxs = []

            # Slide through the windows one by one
            window_height = np.int(warped.shape[0] / num_windows)
            for w in range(num_windows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = warped.shape[0] - (w + 1) * window_height
                win_y_high = warped.shape[0] - w * window_height
                win_x_low = xcurrent[which] - margin_detect
                win_x_high = xcurrent[which] + margin_detect

                # Draw the window for visualization
                cv2.rectangle(canvas, (win_x_low, win_y_low), (win_x_high, win_y_high),
                              (0, 255, 0), thickness=2)

                # Identify the nonzero pixels in x and y within the window
                good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                             (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
                # Append these indices to the lists
                nonzero_idxs.append(good_inds)

                # If number of good pixels > the threshold, recenter next window on their mean position.
                if len(good_inds) > recenter_threshold[1]:
                    xcurrent[which] = np.int(np.mean(nonzerox[good_inds]))
                    debug and print(w, which, 'updated', xcurrent[which])
                # If number of good pixels < the threshold, recenter next window to base.
                elif len(good_inds) < recenter_threshold[0]:
                    xcurrent[which] = xbase[which]
                    debug and print(w, which, "reverted", xcurrent[which])
                else:
                    debug and print(w, which, 'remained', xcurrent[which])

            # Concatenate the arrays of indices
            nonzero_idx = np.concatenate(nonzero_idxs)

        # Extract line pixel positions
        lane_x = nonzerox[nonzero_idx]
        lane_y = nonzeroy[nonzero_idx]
        # Color the pixels for visualization
        canvas[lane_y, lane_x] = color[which]
        pixels[lane_y, lane_x] = color[which]

        # Fit a second order polynomial on pixel distance
        coeff_pixel = np.polyfit(lane_y, lane_x, deg=2)

        # Fit a second order polynomial on world distance
        coeff_world = np.polyfit(lane_y * ym_per_pixel, lane_x * xm_per_pixel, deg=2)

        # Visualize the fitted curve on the canvas
        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        fitx = poly_value(ploty, coeff_pixel)
        # The equivalent of matplotlib.pyplot.plot(X, Y)
        for x, y in zip(fitx, ploty):
            cv2.circle(canvas, center=(int(x), int(y)), radius=3, color=[255, 255, 0], thickness=2)

        # Draw search window for the tracking mode
        if is_tracking:
            line_window1 = np.array([np.transpose(np.vstack([fitx - margin_track, ploty]))])
            line_window2 = np.array([np.flipud(np.transpose(np.vstack([fitx + margin_track, ploty])))])
            line_pts = np.hstack((line_window1, line_window2))
            window_img = np.zeros_like(canvas)
            cv2.fillPoly(window_img, np.int_([line_pts]), (0, 255, 0))
            canvas = cv2.addWeighted(canvas, 1, window_img, 0.3, 0)

        # Generate the polygon points to draw the fitted lane region.
        pts = np.transpose(np.vstack([fitx, ploty]))
        if which == 'right':
            # So that when h-stacked later, the bottom left lane is adjacent to the bottom right lane (U-shape).
            pts = np.flipud(pts)
        region_pts[which] = np.array([pts])  # Don't miss the [] around pts

        # Compute radius of curvature and lane X position where the vehicle is (bottom of the view).
        curv = radius_of_curvature(np.max(ploty) * ym_per_pixel, coeff_world)
        debug and print(which, "curvature", curv)
        lane_radius[which] = curv
        lane_xpos[which] = poly_value(np.max(ploty), coeff_pixel)

        # Update lane info (TODO: perform sanity check after both lanes are detected)
        if lane_infos is not None:
            lane_infos[which].fit_valid = True
            lane_infos[which].fit_coeff = coeff_pixel

    # TODO: sanity check and update lane info

    # Draw the region between left and right lanes.
    cv2.fillPoly(region, np.int_([np.hstack(region_pts.values())]), (0, 255, 0))

    avg_radius = np.mean(list(lane_radius.values()))
    dist_center = (midpoint - np.mean(list(lane_xpos.values()))) * xm_per_pixel

    if img_fname is not None:
        output_img(canvas, os.path.join(output_dir, 'detect-canvas', img_fname))
    return canvas, region, pixels, avg_radius, dist_center


def process_pipeline(img, lane_infos, mtx, dist, output_dir, img_base_fname):
    """
    Image processing pipeline.
    :param img: image pixels array, in BGR color space
    :param lane_infos: lane information for look-ahead filter, reset, and smoothing
    :param mtx: for camera calibration
    :param dist: for camera calibration
    :param output_dir: intermediate output directory
    :param img_base_fname: base filename for this image, None for disabling intermediate output
    :return: annotated output image
    """
    img_size = get_img_size(img)

    # Preprocess image
    undist, masked_color, warped, Minv = preprocess_image(img, mtx, dist, output_dir, img_base_fname)

    # Detect lane
    window, region, pixels, radius, distance = detect_lane(
        warped, lane_infos, num_windows=9, margin_detect=75, margin_track=100, recenter_threshold=(10, 100),
        output_dir=output_dir, img_fname=img_base_fname)

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
    parser.add_argument('--video-file', type=str, required=False, default='./project_video.mp4',
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
            out = process_pipeline(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), None, mtx, dist,
                                   'output_images', os.path.basename(fname))
            output_img(out, os.path.join('output_images', os.path.basename(fname)))

    if x.video_file:
        gen = fname_generator(max_num_frame=10)
        clip = VideoFileClip(x.video_file).subclip(0, 10)
        lane_infos = {'left': LaneInfo(), 'right': LaneInfo()}
        write_clip = clip.fl_image(lambda frame:  # RGB
            process_pipeline(frame, lane_infos, mtx, dist,
                             'output_images_' + os.path.basename(x.video_file), next(gen)))
        write_clip.write_videofile('./project_video_overlay.mp4', audio=False)
