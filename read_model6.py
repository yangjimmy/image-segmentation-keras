# from keras_segmentation.predict import predict
# from curses import raw
from tensorflow.keras.models import Model, model_from_json
import random
import matplotlib.pyplot as plt
import cv2
import six
import numpy as np
import os
import time
import scipy.io
import shutil
import os

#TODO: read in .raw file -> actual files in OCT
    # scaled intensity values
    # E:\YKL\Thorlabs VSCAN Labeling\scripts\functions\loadvscan_v5.m
    # E:\YKL\Thorlabs VSCAN Labeling\scripts\extractBSCAN4Labeling.m
    # E:\YKL\Thorlabs VSCAN Labeling\scripts\extractVSCAN4Labeling.m

def initialize():
    global loaded_model
    json_file = open('C:\\Users\\Computer\\Documents\\GitHub\\image-segmentation-keras\\models\\modelP_unet_vgg_1_epoch.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.output_width = 192
    loaded_model.output_height = 176
    loaded_model.n_classes = 5
    loaded_model.input_height = 352
    loaded_model.input_width = 384
    loaded_model.model_name = ""

    loaded_model.load_weights('C:\\Users\\Computer\\Documents\\GitHub\\image-segmentation-keras\\models\\modelW_unet_vgg_1_epoch.h5')

    global class_colors
    DATA_LOADER_SEED = 0

    random.seed(DATA_LOADER_SEED)
    class_colors = [(random.randint(0, 255), random.randint(
        0, 255), random.randint(0, 255)) for _ in range(10)]


class DataLoaderError(Exception):
    pass

def predict(model=None, inp=None):

    assert (inp is not None)
    assert (type(inp) is np.ndarray),\
        "Input should be the CV image"

    inp = read_bscan(inp)
    inp = preprocess_for_inference(inp)

    assert (len(inp.shape) == 3 or len(inp.shape) == 1 or len(inp.shape) == 4), "Image should be h,w,3 "

    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = get_image_array(inp, input_width, input_height,
                        ordering="channels_last")
    
    pr = model.predict(np.array([x]))[0]
    pr = pr.reshape((output_height,  output_width, n_classes)).argmax(axis=2)

    # seg_img = visualize_segmentation(pr, inp, n_classes=n_classes,
    #                                  colors=colors, overlay_img=overlay_img,
    #                                  show_legends=show_legends,
    #                                  class_names=class_names,
    #                                  prediction_width=prediction_width,
    #                                  prediction_height=prediction_height)

    # if out_fname is not None:
    #     cv2.imwrite(out_fname, seg_img)

    return pr

def get_image_array(image_input,
                    width, height,
                    imgNorm="sub_mean", ordering='channels_first', read_image_type=1):
    """ Load image array from input """

    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif isinstance(image_input, six.string_types):
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_image_array: path {0} doesn't exist"
                                  .format(image_input))
        img = cv2.imread(image_input, read_image_type)
    else:
        raise DataLoaderError("get_image_array: Can't process input type {0}"
                              .format(str(type(image_input))))

    if imgNorm == "sub_and_divide":
        img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
    elif imgNorm == "sub_mean":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = np.atleast_3d(img)

        means = [103.939, 116.779, 123.68]

        for i in range(min(img.shape[2], len(means))):
            img[:, :, i] -= means[i]

        img = img[:, :, ::-1]
    elif imgNorm == "divide":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = img/255.0

    if ordering == 'channels_first':
        img = np.rollaxis(img, 2, 0)
    
    
    return img

def preprocess_for_inference(img):
    X_test_test = []

    resized_img = cv2.equalizeHist(np.clip(cv2.resize(img,(384, 352), interpolation = cv2.INTER_NEAREST),0,255))
    resized_denoised_img = cv2.fastNlMeansDenoising(resized_img,10,7,21)
    resized_denoised_img = cv2.cvtColor(resized_denoised_img,cv2.COLOR_GRAY2RGB)
    X_test_test = resized_denoised_img

    return X_test_test

def run_test(inp_array):
    out = predict(model=loaded_model, inp=inp_array)
    return out


def read_bscan(img_arr):
    raw_file = np.reshape(img_arr,(800,1024))
    raw_file = raw_file[0:400,:]
    raw_file = np.clip(raw_file, 20, 70, out=raw_file)
    raw_file = (raw_file - 20) / 50
    raw_file = raw_file.T

    raw_file_resized = cv2.resize(raw_file, (640,376))
    raw_file_resized = 255 * raw_file_resized

    plt.imshow(raw_file_resized)
    
    return raw_file_resized.astype(np.uint8)


# if __name__ == '__main__':
#     initialize()

#     directory = "E:\\MSR2\\SAVES"
#     output_directory = os.path.join(directory,"segmented_mat")
#     if os.path.exists(output_directory):
#         shutil.rmtree(output_directory)
#     os.mkdir(os.path.join(directory,"segmented_mat"))

        
#     for filename in os.listdir(directory):
#         if filename.endswith(".raw"):
#             full_filepath = os.path.join(directory, filename)
#             full_mat_filepath = os.path.join(output_directory, os.path.splitext(filename)[0]+".mat")
#             with open(full_filepath, 'rb') as f:
#                 raw_file = np.fromfile(f, dtype=np.float32)
#             out = run_test(raw_file)
#             scipy.io.savemat(full_mat_filepath, {'img': out})
#         else:
#             continue
    

# if __name__ == '__main__':
#     image_loc = "C:\\Users\\Computer\\Documents\\GitHub\\image-segmentation-keras\\VSCAN_0027_190.png"
#     img = cv2.imread(image_loc, cv2.IMREAD_GRAYSCALE)
#     print(img.shape)
#     plt.imshow(img)
#     plt.show()

if __name__ == '__main__':
    with open("E:\\MSR2\\SAVES\\20240417\\BSCAN_65267268.raw", 'rb') as f:
        raw_file = np.fromfile(f, dtype=np.float32)
    initialize()
    # for i in range(10000):
    # raw_file_ = read_bscan(raw_file)
    out = run_test(raw_file)
    scipy.io.savemat("E:\\MSR2\\SAVES\\20240417\\BSCAN_65267268.mat", {'img': out})

    plt.figure()
    plt.imshow(out)
    # plt.figure()
    # plt.imshow(cv2.imread("C:\\Users\\Computer\\Documents\\GitHub\\image-segmentation-keras\\VSCAN_0027_190.png"))
    plt.show()

    # labels: 1 = cornea; 2 = sclera; 3 = retina; 4 = background


# if __name__ == '__main__':
    # initialize()
    # start = time.time()
        
    # out = predict(model=loaded_model, inp="C:\\Users\\Computer\\Documents\\GitHub\\image-segmentation-keras\\VSCAN_0027_190.png", out_fname=None,
    #             checkpoints_path="tmp\\", overlay_img=False,
    #             class_names=None, show_legends=False, colors=class_colors,
    #             prediction_width=None, prediction_height=None,
    #                 read_image_type=1)
    # print(type(out[0][0]))
    # print("time elapsed: {}".format(time.time()-start))

    # # show figure
    # plt.figure()
    # plt.imshow(out)
    # plt.figure()
    # plt.imshow(cv2.imread("C:\\Users\\Computer\\Documents\\GitHub\\image-segmentation-keras\\VSCAN_0027_190.png"))
    # plt.show()

    # start = time.time()
    # out = predict(model=loaded_model, inp="C:\\Users\\Computer\\Documents\\GitHub\\image-segmentation-keras\\VSCAN_0027_190.png", out_fname=None,
    #             checkpoints_path="tmp\\", overlay_img=False,
    #             class_names=None, show_legends=False, colors=class_colors,
    #             prediction_width=None, prediction_height=None,
    #             read_image_type=1)
    # print("time elapsed: {}".format(time.time()-start))
