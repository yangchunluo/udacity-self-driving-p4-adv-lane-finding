import numpy as np
import cv2
import glob
import argparse
import os
import pickle


def calibrate_camera(image_dir, dimensions, output_corners_found, output_undistort):
    """
    Calibrate camera based on a set of chessboard images.

    :param image_dir: directory of chessboard images
    :param dimensions: chessboard size
    :param output_corners_found: directory to output images with corners found
    :param output_undistort: directory to output undistorted images
    :return: camera calibration settings
    """

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0).
    objp = np.zeros((dimensions[0] * dimensions[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:dimensions[0], 0:dimensions[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space.
    imgpoints = []  # 2d points in image plane.

    # Loop through all the calibration images.
    images = glob.glob(os.path.join(image_dir, 'calibration*.jpg'))
    image_size = None
    for idx, fname in enumerate(images):
        # Read in a chessboard image.
        img = cv2.imread(fname)
        # Assume all images are of the same pixel dimension.
        if image_size is None:
            image_size = (img.shape[1], img.shape[0])
        # Convert to gray scale.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners.
        ret, corners = cv2.findChessboardCorners(gray, dimensions, None)

        # If found, add object points, image points.
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Output images with corners found.
            if output_corners_found is not None:
                cv2.drawChessboardCorners(img, dimensions, corners, ret)
                out_fname = os.path.join(output_corners_found, os.path.basename(fname))
                print(out_fname)
                cv2.imwrite(out_fname, img)

    # Calibrate camera.
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

    # Output undistorted images.
    if output_undistort:
        for fname in images:
            img = cv2.imread(fname)
            undist = cv2.undistort(img, mtx, dist, None, mtx)
            out_fname = os.path.join(output_undistort, os.path.basename(fname))
            print(out_fname)
            cv2.imwrite(out_fname, undist)

    return mtx, dist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str, required=False, default='./camera_cal',
                        help='Directory of the chessboard calibration images')
    parser.add_argument('--chessboard-dim', type=str, required=False, default='9,6',
                        help='Comma separated 2-tuple for chessboard dimensions (width, height)')
    parser.add_argument('--output-corners-found', type=str, required=False,
                        help='Directory to optionally output the images with corners found')
    parser.add_argument('--output-undistort', type=str, required=False, default='output_images/calibrated',
                        help='Directory to optionally output the undistorted images')
    parser.add_argument('--output-file', type=str, required=False, default='./calibration-params.p',
                        help='Output pickle file for the calibration parameters')
    x = parser.parse_args()

    chessboard_dim = tuple([int(d) for d in x.chessboard_dim.split(',')])
    mtx, dist = calibrate_camera(x.image_dir, chessboard_dim, x.output_corners_found, x.output_undistort)

    # Save the output file
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open(x.output_file, "wb"))