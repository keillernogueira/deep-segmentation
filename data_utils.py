import os

import imageio
import numpy as np
from skimage import transform


def load_image(image_path):  # os.path.join(dataset_path, image_name)
    try:
        img = imageio.imread(image_path)
    except IOError:
        raise IOError("Could not open file: ", image_path)

    return np.asarray(img)


def create_distrib(labels, crop_size, stride_size, num_classes, return_all=False):
    instances = [[[] for i in range(0)] for i in range(num_classes)]
    counter = num_classes * [0]
    binc = np.zeros((num_classes, num_classes))  # cumulative bincount for each class

    for k in range(0, len(labels)):
        w, h = labels[k].shape
        for i in range(0, w, stride_size):
            for j in range(0, h, stride_size):
                cur_map = k
                cur_x = i
                cur_y = j
                patch_class = labels[cur_map][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]

                if len(patch_class) != crop_size and len(patch_class[0]) != crop_size:
                    cur_x = cur_x - (crop_size - len(patch_class))
                    cur_y = cur_y - (crop_size - len(patch_class[0]))
                    patch_class = labels[cur_map][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]
                elif len(patch_class) != crop_size:
                    cur_x = cur_x - (crop_size - len(patch_class))
                    patch_class = labels[cur_map][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]
                elif len(patch_class[0]) != crop_size:
                    cur_y = cur_y - (crop_size - len(patch_class[0]))
                    patch_class = labels[cur_map][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]

                assert patch_class.shape == (crop_size, crop_size), \
                    "Error create_distrib: Current patch size is " + str(len(patch_class)) + "x" + str(len(patch_class[0]))

                count = np.bincount(patch_class.astype(int).flatten(), minlength=2)
                if count[1] != 0:
                    instances[1].append((cur_map, cur_x, cur_y, np.bincount(patch_class.flatten())))
                    counter[1] += 1
                    binc[1] += count
                else:
                    instances[0].append((cur_map, cur_x, cur_y, np.bincount(patch_class.flatten())))
                    counter[0] += 1
                    binc[0] += count

    for i in range(len(counter)):
        print('Class ' + str(i) + ' has length ' + str(counter[i]) + ' - ' + np.array_str(binc[i]).replace("\n", ""))

    if return_all:
        return np.asarray(instances[0] + instances[1]), \
               np.concatenate((np.zeros(len(instances[0])), np.ones(len(instances[1]))))
    else:
        # empty because does not make sense
        return np.asarray(instances[1]), np.asarray([])


def normalize_images(data, _mean, _std):
    for i in range(len(_mean)):
        data[:, :, i] = np.subtract(data[:, :, i], _mean[i])
        data[:, :, i] = np.divide(data[:, :, i], _std[i])


def compute_image_mean(data):
    _mean = np.mean(np.mean(np.mean(data, axis=0), axis=0), axis=0)
    _std = np.std(np.std(np.std(data, axis=0, ddof=1), axis=0, ddof=1), axis=0, ddof=1)

    return _mean, _std


def calculate_mean_and_std(data, distrib, crop_size):
    all_patches = []

    for i in range(len(distrib)):
        cur_map = distrib[i][0]
        cur_x = distrib[i][1]
        cur_y = distrib[i][2]
        patch = data[cur_map][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]

        if len(patch[0]) != crop_size or len(patch[1]) != crop_size:
            raise NotImplementedError("Error! Current patch size: " +
                                      str(len(patch)) + "x" + str(len(patch[0])))

        all_patches.append(patch)

    # remaining images
    return compute_image_mean(np.asarray(all_patches))


def create_or_load_statistics(data, distrib, crop_size, stride_size, output_path):
    # create mean, std from training
    if os.path.isfile(os.path.join(output_path, 'crop_' + str(crop_size) + '_stride_' +
                      str(stride_size) + '_mean.npy')):
        print("Loading mean and std")
        _mean = np.load(os.path.join(output_path, 'crop_' + str(crop_size) + '_stride_' +
                                     str(stride_size) + '_mean.npy'), allow_pickle=True)
        _std = np.load(os.path.join(output_path, 'crop_' + str(crop_size) + '_stride_' +
                                    str(stride_size) + '_std.npy'), allow_pickle=True)
    else:
        print("Calculating mean and std")
        _mean, _std = calculate_mean_and_std(data, distrib, crop_size)
        np.save(os.path.join(output_path, 'crop_' + str(crop_size) + '_stride_' +
                str(stride_size) + '_mean.npy'), _mean)
        np.save(os.path.join(output_path, 'crop_' + str(crop_size) + '_stride_' +
                str(stride_size) + '_std.npy'), _std)
    print(_mean, _std)
    return _mean, _std


def data_augmentation(img, label=None):
    rand_fliplr = np.random.random() > 0.50
    rand_flipud = np.random.random() > 0.50
    rand_rotate = np.random.random()

    if rand_fliplr:
        img = np.fliplr(img)
        label = np.fliplr(label)
    if rand_flipud:
        img = np.flipud(img)
        label = np.flipud(label)

    if rand_rotate < 0.25:
        img = transform.rotate(img, 270, order=1, preserve_range=True)
        label = transform.rotate(label, 270, order=0, preserve_range=True)
    elif rand_rotate < 0.50:
        img = transform.rotate(img, 180, order=1, preserve_range=True)
        label = transform.rotate(label, 180, order=0, preserve_range=True)
    elif rand_rotate < 0.75:
        img = transform.rotate(img, 90, order=1, preserve_range=True)
        label = transform.rotate(label, 90, order=0, preserve_range=True)

    img = img.astype(np.float32)
    label = label.astype(np.int64)

    return img, label
