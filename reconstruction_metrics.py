import cv2
import numpy as np
from skimage.metrics import structural_similarity
from tqdm import tqdm

from video_util import load_data


# Typically result values are anywhere between 30 and 50 for compression, where higher is better.
# If the images significantly differ you'll get much lower ones like 15 and so.
def psnr(basic_video, modified_video, data_type):
    basic_video_images = load_data(basic_video, data_type)
    modified_video_images = load_data(modified_video, data_type)
    psnr_results = []
    assert len(basic_video_images) == len(modified_video_images)
    for i in range(len(basic_video_images)):
        psnr_results.append(cv2.PSNR(basic_video_images[i], modified_video_images[i]))
    return psnr_results, sum(psnr_results) / len(psnr_results)


def mse(basic_video, modified_video, data_type):
    basic_video_images = load_data(basic_video, data_type)
    modified_video_images = load_data(modified_video, data_type)
    mse_results = []
    assert len(basic_video_images) == len(modified_video_images)
    for i in range(len(basic_video_images)):
        difference_array = np.subtract(basic_video_images[i], modified_video_images[i])
        squared_array = np.square(difference_array)
        mse_results.append(squared_array.mean())
    return mse_results, sum(mse_results) / len(mse_results)


# value is between zero and one, where one corresponds to perfect fit
def ssim(basic_video, modified_video, data_type):
    basic_video_images = load_data(basic_video, data_type)
    modified_video_images = load_data(modified_video, data_type)
    ssim_results = []
    assert len(basic_video_images) == len(modified_video_images)
    for i in tqdm(range(len(basic_video_images))):
        ssim_results.append(
            structural_similarity(basic_video_images[i], modified_video_images[i], multichannel=True)
        )
    return ssim_results, sum(ssim_results) / len(ssim_results)


if __name__ == '__main__':
    path1 = 'path_to_first_video_frames_or_video'
    path2 = 'path_to_second_video_frames_or_video'
    psnr_all_results, psnr_avg = psnr(path1, path2, data_type='hdr')
    mse_all_results, mse_avg = mse(path1, path2, data_type='hdr')
    ssim_all_results, ssim_avg = ssim(path1, path2, data_type='hdr')