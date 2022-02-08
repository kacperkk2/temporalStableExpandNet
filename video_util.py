import os
from os.path import isfile, join
from shutil import copyfile

import cv2
import numpy as np

from util import map_range


def load_data(path, data_type):
    if data_type == "ldr":
        return load_ldr_frames(path)
    elif data_type == "hdr":
        return load_hdr_frames(path)
    elif data_type == "video":
        return load_video_frames(path)
    else:
        raise AssertionError("this data type is not allowed")


def load_ldr_frames(path):
    files = get_files_from_dir(path, format=".png")
    files_list = []
    for file in files:
        image = cv2.imread(file)
        files_list.append(image)
    if len(files_list) == 0:
        raise RuntimeError(f'Cannot load files from: {path}')
    return files_list


def load_hdr_frames(path):
    files = get_files_from_dir(path, format=".exr")
    files_list = []
    for file in files:
        image = cv2.imread(file, flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR)
        image = map_range(image, 0, 1)
        files_list.append(image)
    if len(files_list) == 0:
        raise RuntimeError(f'Cannot load files from: {path}')
    return files_list


def load_video_frames(video_path):
    cap_in = cv2.VideoCapture(video_path)
    n_frames = cap_in.get(cv2.CAP_PROP_FRAME_COUNT)
    images = []
    while cap_in.isOpened():
        perc = cap_in.get(cv2.CAP_PROP_POS_FRAMES) * 100 / n_frames
        print('\rLoading video: {0:.2f}%'.format(perc), end='')
        ret, loaded = cap_in.read()
        if loaded is None:
            break
        images.append(loaded)
    cap_in.release()
    print("")
    return images


def get_files_from_dir(dir, format):
    files = []
    for p, d, f in sorted(os.walk(dir)):
        for file in f:
            if file.endswith(format):
                files.append(os.path.join(p, file))
    files = sorted(files)
    if len(files) == 0:
        raise AssertionError('No files in directory!')
    return files


def convert_tiff_to_exr(dir):
    files = get_files_from_dir(dir, '.tiff')
    for file in files:
        input = cv2.imread(file, -1).astype(np.uint16)
        input = gamma_correction(input, gamma=0.2).astype(np.uint16)
        exr_name = file.replace('tiff', 'exr')
        cv2.imwrite(exr_name, input.astype(np.float32))


def to_frames(video_name, dest_dir):
    vidcap = cv2.VideoCapture(video_name)
    print(f"videocap opened: {vidcap.isOpened()}")
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(os.path.join(dest_dir, "frame%04d.jpg" % count), image)
        success, image = vidcap.read()
        count += 1


def to_video(in_dir, video_name_out):
    files = [f for f in os.listdir(in_dir) if isfile(join(in_dir, f)) and (f.endswith(".png") or f.endswith(".jpg"))]
    files.sort()
    first_frame = cv2.imread(os.path.join(in_dir, files[0]))
    height, width, layers = first_frame.shape
    size = (width, height)
    print(f"Frames size: {size}")
    FPS = 30
    out = cv2.VideoWriter(os.path.join(in_dir, video_name_out), cv2.VideoWriter_fourcc(*'DIVX'), FPS, size)
    for file in files:
        frame = cv2.imread(os.path.join(in_dir, file))
        out.write(frame)
    out.release()


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 65535) ** invGamma) * 65535 for i in np.arange(0, 65536)]).astype("uint16")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def copy_exr_to_dir(from_dir, to_dir, format='.exr'):
    files = get_files_from_dir(from_dir, format)
    for file in files:
        copyfile(file, to_dir + file.split('/')[-2] + '_' + file.split('/')[-1])


def gamma_correction(img: np.ndarray, gamma: float=1.0):
    igamma = 1.0 / gamma
    imin, imax = img.min(), img.max()

    img_c = img.copy()
    img_c = ((img_c - imin) / (imax - imin)) ** igamma
    img_c = img_c * (imax - imin) + imin
    return img_c


def tone_map_reinhard(in_dir, out_dir):
    if not os.path.exists(f'{out_dir}/reinhard'):
        os.mkdir(f'{out_dir}/reinhard')
    files = get_files_from_dir(in_dir, format=".exr")
    for iter, file in enumerate(files):
        image = cv2.imread(file, flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR)
        tmo_img = reinhard(image)
        name = file.split("/")[-1][:-4]
        cv2.imwrite(f'{out_dir}/reinhard/{name}.png', (tmo_img * 255).astype(int))


def reinhard(img, intensity=-1.0, light_adapt=0.8, color_adapt=0.0, gamma=2.0):
    op = cv2.createTonemapReinhard(
        gamma=gamma,
        intensity=intensity,
        light_adapt=light_adapt,
        color_adapt=color_adapt
    )
    return op.process(img)