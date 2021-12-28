import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from math import sqrt
import sys
import getopt
import time
from numba import jit
#import scipy.interpolate as interp

IMWRITE_TIFF_COMPRESSION = 259
IMWRITE_TIFF_COMPRESSION_NONE = 1
IMWRITE_TIFF_RESUNIT_INCH = 2

start_time = time.time()

def log(msg):
    dt = time.time() - start_time
    print('[{0:.2f}] {1}'.format(dt, msg))

def extendPattern(corners_mtx, img_size):
    img_w, img_h = img_size
    corners_ext = corners_mtx

    # extend left and right
    corners_ext_left = []
    corners_ext_right = []
    can_extend_left = False
    can_extend_right = False
    for row in range(corners_ext.shape[0]):
        # linear extrapolation
        c1_right = corners_ext[row, -2, 0]
        c2_right = corners_ext[row, -1, 0]
        c3_right = c2_right + (c2_right - c1_right)
        if c2_right[0] < img_w and c2_right[0] > 0:
            can_extend_right = True
        corners_ext_right.append(np.array([c3_right]))

        # extrapolate by spline
        # spline_k = 4
        # if corners_ext[row, -1, 0, 0] < img_w and corners_ext[row, -1, 0, 0] > 0:
        #     can_extend_right = True
        # c_right = corners_ext[row, -(spline_k + 1):, 0]
        # spl_right = interp.splprep([c_right[:,0], c_right[:,1]], k=spline_k, u=list(range(-(spline_k+1), 0)))
        # splev_right = interp.splev(0, spl_right[0])
        # c_ext_right = np.array([float(splev_right[0]), float(splev_right[1])], dtype=np.float32)
        # corners_ext_right.append(np.array([c_ext_right]))

        c1_left = corners_ext[row, 1, 0]
        c2_left = corners_ext[row, 0, 0]
        c3_left = c2_left + (c2_left - c1_left)
        if c2_left[0] < img_w and c2_left[0] > 0:
            can_extend_left = True
        corners_ext_left.append(np.array([c3_left]))

        # extrapolate by spline
        # if corners_ext[row, 0, 0, 0] < img_w and corners_ext[row, 0, 0, 0] > 0:
        #     can_extend_left = True
        # c_left = corners_ext[row, 0:(spline_k+1), 0]
        # spl_left = interp.splprep([c_left[:,0], c_left[:,1]], k=spline_k, u=list(range(1,spline_k+2)))
        # splev_left = interp.splev(0, spl_left[0])
        # c_ext_left = np.array([float(splev_left[0]), float(splev_left[1])], dtype=np.float32)
        # corners_ext_left.append(np.array([c_ext_left]))

    if can_extend_right:
        ce_right = np.reshape(np.array(corners_ext_right), (corners_ext.shape[0], 1, 1, 2))
        corners_ext = np.hstack((corners_ext, ce_right))
    if can_extend_left:
        ce_left = np.reshape(np.array(corners_ext_left), (corners_ext.shape[0], 1, 1, 2))
        corners_ext = np.hstack((ce_left, corners_ext))

    # extend top and bottom
    corners_ext_top = []
    corners_ext_bot = []
    can_extend_bot = False
    can_extend_top = False
    for col in range(corners_ext.shape[1]):
        c1_bot = corners_ext[-2, col, 0]
        c2_bot = corners_ext[-1, col, 0]
        c3_bot = c2_bot + (c2_bot - c1_bot)
        # if c3_bot[1] > img_h or c3_bot[1] < 0:
        #     can_extend_bot = False
        if c2_bot[1] < img_h and c2_bot[1] > 0:
            can_extend_bot = True
        corners_ext_bot.append(np.array([c3_bot]))

        c1_top = corners_ext[1, col, 0]
        c2_top = corners_ext[0, col, 0]
        c3_top = c2_top + (c2_top - c1_top)
        # if c3_top[1] < 0 or c3_top[1] > img_h:
        #     can_extend_top = False
        if c2_top[1] < img_h and c2_top[1] > 0:
            can_extend_top = True
        corners_ext_top.append(np.array([c3_top]))

    if can_extend_bot:
        ce_bot = np.reshape(np.array(corners_ext_bot), (1, corners_ext.shape[1], 1, 2))
        corners_ext = np.vstack((corners_ext, ce_bot))
    if can_extend_top:
        ce_top = np.reshape(np.array(corners_ext_top), (1, corners_ext.shape[1], 1, 2))
        corners_ext = np.vstack((ce_top, corners_ext))

    return can_extend_bot or can_extend_top or can_extend_right or can_extend_left, corners_ext

@jit(nopython=True)
def mapsxy(dst_w, dst_h, pattern_w, pattern_h, corners):
    mapx = np.zeros(shape=(dst_h, dst_w), dtype=np.float32)
    mapy = np.zeros(shape=(dst_h, dst_w), dtype=np.float32)
    # for every destination image pixel
    for dst_y in range(dst_h):
        for dst_x in range(dst_w):
            # coords in pattern coord system (square side = 1)
            square_x = float(dst_x) * (pattern_w - 1) / float(dst_w)
            square_y = float(dst_y) * (pattern_h - 1) / float(dst_h)
            # pattern square coords
            square_x_idx = int(square_x)
            square_y_idx = int(square_y)
            # coords inside of square (0...1)
            square_x_frac = square_x - square_x_idx
            square_y_frac = square_y - square_y_idx
            # square vertex coords in source image
            c00 = corners[square_y_idx * pattern_w + square_x_idx, 0]
            c01 = corners[square_y_idx * pattern_w + square_x_idx + 1, 0]
            c10 = corners[(square_y_idx + 1) * pattern_w + square_x_idx, 0]
            c11 = corners[(square_y_idx + 1) * pattern_w + square_x_idx + 1, 0]

            if square_x_frac + square_y_frac <= 1:
                # we are above diagonal, use basis c00, c01, c10
                # src_v = np.dot([[c01[0] - c00[0], c10[0] - c00[0]],
                #                 [c01[1] - c00[1], c10[1] - c00[1]]], [square_x_frac, square_y_frac])
                # x = c00[0] + src_v[0]
                # y = c00[1] + src_v[1]
                x = c00[0] + (c01[0] - c00[0]) * square_x_frac + (c10[0] - c00[0]) * square_y_frac
                y = c00[1] + (c01[1] - c00[1]) * square_x_frac + (c10[1] - c00[1]) * square_y_frac
            else:
                # we are below diagonal, use basis c11, c01, c10
                # src_v = np.dot([[c10[0] - c11[0], c01[0] - c11[0]],
                #                 [c10[1] - c11[1], c01[1] - c11[1]]], [1-square_x_frac, 1-square_y_frac])

                # x = c11[0] + src_v[0]
                # y = c11[1] + src_v[1]
                x = c11[0] + (c10[0] - c11[0]) * (1 - square_x_frac) + (c01[0] - c11[0]) * (1 - square_y_frac)
                y = c11[1] + (c10[1] - c11[1]) * (1 - square_x_frac) + (c01[1] - c11[1]) * (1 - square_y_frac)

            mapx[dst_y, dst_x] = x
            mapy[dst_y, dst_x] = y
    return (mapx, mapy)

def dewarp(pattern_file, pattern_w, pattern_h, picture_file, dewarped_file, corners_file=None, gamma=2.2, square_size_mm=10):
    log('dewrap {0} with {1}'.format(picture_file, pattern_file))

    log('read {0}'.format(pattern_file))
    # read 8 bit color or grayscale image
    pattern_img = cv.imread(pattern_file, cv.IMREAD_COLOR)
    assert(pattern_img.dtype == np.uint8) # 8-bit
    log('{0}'.format(pattern_img.dtype))
    log('convert to grayscale')
    pattern_img_gray = cv.cvtColor(pattern_img, cv.COLOR_BGR2GRAY)
    # if gray.dtype != np.dtype(np.uint8):
    #     log('convert to 8 bit')
    #     gray = np.uint8(gray/256.0 + 0.5)

    pattern_size = (pattern_w, pattern_h)

    # figure1 = plt.figure()
    # plt.imshow(pattern_img)
    h = pattern_img.shape[0]
    w = pattern_img.shape[1]

    # Source chessboard view must be an 8-bit grayscale or color image
    log('find chessboard corners')
    found, corners = cv.findChessboardCorners(pattern_img_gray, pattern_size, None)

    if found == 0:
        log('failed')
        return

    term_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    log('find chessboard corners subpixel coords')
    # single-channel, 8-bit or float image.
    corners = cv.cornerSubPix(pattern_img_gray, corners, (11,11), (-1,-1), term_criteria)

    # make first corner be upper left
    if corners[0, 0, 0] > corners[pattern_w - 1, 0, 0]:
        log('pattern rotated by 180 deg')
        corners = np.flip(corners, 0)
    log('extend pattern')
    while True:
        corners_mtx = np.reshape(corners, (pattern_h, pattern_w, 1, 2))
        extended, corners_mtx = extendPattern(corners_mtx, (w, h))
        pattern_w = corners_mtx.shape[1]
        pattern_h = corners_mtx.shape[0]
        pattern_size = (pattern_w, pattern_h)
        corners = np.reshape(corners_mtx, (corners_mtx.shape[0] * corners_mtx.shape[1], 1, 2))
        if extended:
            continue
        break

    # render corners
    if corners_file is not None:
        log('render corners')
        # Destination image must be an 8-bit color image.
        if len(pattern_img.shape) == 1:
            corners_img = cv.cvtColor(pattern_img, cv.COLOR_GRAY2BGR)
        else:
            corners_img = pattern_img.copy()
        cv.drawChessboardCorners(corners_img, pattern_size, corners, found)
        log('write corners to {0}'.format(corners_file))
        cv.imwrite(corners_file, corners_img, params=[IMWRITE_TIFF_COMPRESSION, IMWRITE_TIFF_COMPRESSION_NONE])

    # calc destination image size

    # diagonal
    c00 = corners[0, 0]
    c01 = corners[pattern_w - 1, 0]
    c11 = corners[-1, 0]
    c10 = corners[-pattern_w, 0]
    diag1 = sqrt((c11[0] - c00[0])**2 + (c11[1] - c00[1])**2)
    diag2 = sqrt((c01[0] - c10[0])**2 + (c01[1] - c10[1])**2)
    scale_diag = max(diag1, diag2) / sqrt((pattern_w - 1)**2 + (pattern_h - 1)**2)

    # sides
    dst_top = sqrt((c01[0] - c00[0])**2 + (c01[1] - c00[1])**2)
    scale_top = dst_top / (pattern_w - 1)
    dst_bottom = sqrt((c11[0] - c10[0])**2 + (c11[1] - c10[1])**2)
    scale_bottom = dst_bottom / (pattern_w - 1)
    dst_left = sqrt((c10[0] - c00[0])**2 + (c10[1] - c00[1])**2)
    scale_left = dst_left / (pattern_h - 1)
    dst_right = sqrt((c11[0] - c01[0])**2 + (c11[1] - c01[1])**2)
    scale_right = dst_right / (pattern_h - 1)

    # scale of output image px/square
    scale_pattern = max(scale_diag, scale_top, scale_bottom, scale_left, scale_right)
    # size of output image
    dst_w = round((pattern_w - 1) * scale_pattern)
    dst_h = round((pattern_h - 1) * scale_pattern)

    # cals maps for remap
    log('calc maps')
    mapx, mapy = mapsxy(dst_w, dst_h, pattern_w, pattern_h, corners)

    if picture_file is not None:
        log('read image {0}'.format(picture_file))
        # color & grayscale, 8 & 16 bit
        picture_img = cv.imread(picture_file, cv.IMREAD_COLOR | cv.IMREAD_ANYDEPTH)
        log('{0}'.format(picture_img.dtype))
    else:
        picture_img = pattern_img

    # log("convert to linear float")
    # if picture_img.dtype == np.uint8:
    #     picture_img_fl = (picture_img / np.float32(255))**gamma
    # elif picture_img.dtype == np.uint16:
    #     picture_img_fl = (picture_img / np.float32(65535))**gamma
    # else:
    #     log("image must be 8 or 16 bit")
    #     return
    # log("min={0}, max={1}".format(np.min(picture_img_fl), np.max(picture_img_fl)))

    # log('remap')
    # dst_fl = cv.remap(picture_img_fl, mapx, mapy, cv.INTER_CUBIC)
    # log("min={0}, max={1}".format(np.min(dst_fl), np.max(dst_fl)))

    # log("convert to uint with gamma")
    # dst_fl = np.clip(dst_fl, 0, 1)
    # if picture_img.dtype == np.uint8:
    #     dst = np.uint8(dst_fl**(1.0/gamma) * 255 + 0.5)
    # elif picture_img.dtype == np.uint16:
    #     dst = np.uint16(dst_fl**(1.0/gamma) * 65535 + 0.5)
    # else:
    #     assert(False)

    log('remap')
    dst = cv.remap(picture_img, mapx, mapy, cv.INTER_CUBIC)

    log('write {0}'.format(dewarped_file))
    # libtiff flags: https://gitlab.com/libtiff/libtiff/-/blob/master/libtiff/tiff.h
    dpi = round(25.4 * scale_pattern / square_size_mm)
    # TODO icc profile. but how?
    cv.imwrite(dewarped_file, dst, params=[
        IMWRITE_TIFF_COMPRESSION, IMWRITE_TIFF_COMPRESSION_NONE,
        cv.IMWRITE_TIFF_RESUNIT, IMWRITE_TIFF_RESUNIT_INCH,
        cv.IMWRITE_TIFF_XDPI, dpi,
        cv.IMWRITE_TIFF_YDPI, dpi])

    log('done')

    # figure2 = plt.figure()
    # plt.imshow(corners_img)

    # figure3 = plt.figure()
    # plt.imshow(dst)
    # plt.show()

def usage():
    print("dewarp_photo.py [-g <grid_image>] [-s square_size_mm] <chessboard_image> <width> <height> <input_image> <output_image>")

def batch(dir):
    pass

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "g:s:G:h")
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)
    if len(args) != 5:
        usage()
        sys.exit(2)
    chessboard_file = args[0]
    pattern_w = int(args[1])
    pattern_h = int(args[2])
    image_file = args[3]
    output_file = args[4]
    grid_file = None
    gamma = 2.2
    square_size_mm = None

    for opt, arg in opts:
        if opt == "-g":
            grid_file = arg
        elif opt == "-s":
            square_size_mm = float(arg)
        elif opt == "-G":
            log("WARNING: gamma not implemented")
            gamma = float(arg)
        elif opt == "-h":
            usage()
            sys.exit()
        else:
            assert False, "unhandled option"

    if square_size_mm is None:
        log("WARNING: using default chessboard square size 10 mm")
        square_size_mm = 10

    dewarp(chessboard_file, pattern_w, pattern_h, image_file, output_file, corners_file=grid_file, gamma=gamma, square_size_mm=square_size_mm)

if __name__ == "__main__":
    main()



