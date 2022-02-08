import numpy as np
from scipy.ndimage import gaussian_filter1d

from video_util import load_data


def apply_temporal_gauss(struct):
    for i in range(struct.shape[0]):
        struct[i] = gaussian_filter1d(struct[i], sigma=5)


def smoothness(basic_video, modified_video, data_type):
    basic_video_images = load_data(basic_video, data_type)
    basic_video_images = np.array(basic_video_images)
    gauss = np.transpose(basic_video_images, axes=(2,1,3,0))
    original = np.copy(gauss)
    print(f'Applying temporal gauss on basic video frames...')
    apply_temporal_gauss(gauss)
    basic_video_final_sum = np.sum(np.square(original - gauss))

    stability_video_images = load_data(modified_video, data_type)
    stability_video_images = np.array(stability_video_images)

    assert len(stability_video_images) == len(basic_video_images)
    gauss = np.transpose(stability_video_images, axes=(2,1,3,0))
    original = np.copy(gauss)
    print(f'Applying temporal gauss on stability video frames...')
    apply_temporal_gauss(gauss)
    stability_video_final_sum = np.sum(np.square(original - gauss))

    return np.sqrt(basic_video_final_sum / stability_video_final_sum)


if __name__ == '__main__':
    path1 = 'path_to_first_video_frames_or_video'
    path2 = 'path_to_second_video_frames_or_video'
    S = smoothness(path1, path2, data_type='hdr')

    print(f'Smoothness = {S}')
    if S < 1:
        print('Video passed as second parameter is LESS smooth than video passed as first parameter')
    else:
        print('Video passed as second parameter is MORE smooth than video passed as first parameter')