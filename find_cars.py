import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from feature_tools import *
import bbox_filter


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler,
              color_space, orient, pix_per_cell, cell_per_block,
              spatial_size, hist_bins):

    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    # img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale),
                                                       np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    bboxes = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            combined_feature = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
            test_features = X_scaler.transform(combined_feature)
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                box = ((xbox_left, ytop_draw + ystart),
                       (xbox_left + win_draw, ytop_draw + win_draw + ystart))
                # cv2.rectangle(draw_img, *box, (0, 0, 255), 6)
                bboxes.append(box)

    # return draw_img
    return bboxes


if __name__ == '__main__':

    # load a pe-trained svc model from a serialized (pickle) file
    dist_pickle = pickle.load(open("svc_pickle.p", "rb" ) )

    # get attributes of our svc object
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    color_space = dist_pickle["color_space"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]

    # print(color_space, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    print(dist_pickle)

    # Parameters for sliding window
    ystart = 400
    scale_height_list = [
        # (scale_ratio, height), where yend = ystart + height
        (1.0, 96), (1.12, 112), (1.25, 128), (1.5, 192), (2.0, 256), (2.5, 256), (3.0, 256), (4.0, 256)
    ]

    def find_car_multiscale(img, ystart, scale_height_list):
        all_bboxes = []
        for scale, height in scale_height_list:
            ystop = ystart + height
            bboxes = find_cars(img, ystart, ystop, scale,
                               svc, X_scaler, color_space, orient,
                               pix_per_cell, cell_per_block, spatial_size, hist_bins)
            all_bboxes.extend(bboxes)
        return all_bboxes

    def filter_and_draw_bboxes(img, bboxes, n_box_min_thr, enable_filter=True):
        if enable_filter:
            img, heatmap = bbox_filter.draw_filtered_bbox(img, bboxes, n_box_min_thr)
        else:
            for bbox in bboxes:
                cv2.rectangle(img, bbox[0], bbox[1], (255, 0, 0), 4)  # Draw the box
        return

    def process_test_images(n_box_min_thr=0):
        for img_path in glob.glob('test_images/*.jpg'):
            print('processing: ', img_path)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            bboxes = find_car_multiscale(img, ystart, scale_height_list)
            out_img = filter_and_draw_bboxes(img, bboxes, n_box_min_thr)

            out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join('output_images', os.path.basename(img_path)), out_img)
        return

    def process_test_video(n_frame_history=1, n_box_min_thr=0, enable_filter=False):
        import collections
        from itertools import chain
        bbox_history = collections.deque(maxlen=n_frame_history)  # use like ring buffer

        def pipeline_func(img):
            """ Precondition: color_space of img is RGB """
            bboxes = find_car_multiscale(img, ystart, scale_height_list)
            bbox_history.append(bboxes)
            all_boxes = list(chain.from_iterable(bbox_history))  # flatten
            filter_and_draw_bboxes(img, all_boxes, n_box_min_thr, enable_filter)
            return img

        from moviepy.editor import VideoFileClip
        clip_name = 'project_video.mp4'
        # clip_name = 'test_video.mp4'
        clip1 = VideoFileClip(clip_name)  # .subclip(25, 25.3)
        out_clip = clip1.fl_image(pipeline_func)
        out_clip.write_videofile('output_videos/' + clip_name, audio=False)
        return

    # process_test_images(n_box_min_thr=0)
    process_test_video(4, n_box_min_thr=60)
    # process_test_video(1, n_box_min_thr=0)
