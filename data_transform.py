#############
# data transforms for neural network transform, from a 2D image to 4D blob
#############
import numpy as np

NEW_X = np.newaxis
def image_transform(img, method='norm', center_crop=None):
    """ transform image
    img: input image
    center_crop: [l, r, t, b]
    """
    if 'norm' == method:
        img = np.float32(img) / 255.0
        img -= 0.5

    if center_crop:
        height, width = img.shape[:2]
        up = height / 2 - center_crop / 2
        left = width / 2 - center_crop / 2
        img = img[up:up + center_crop,
                  left:left + center_crop]

    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, axis=0)  # (1, c, h, w)
    return img


def score_transform(score):
    score = np.transpose(score, [2, 0, 1])
    score = np.expand_dims(score, axis=0)  # (1, c, h, w)
    return score


def mask_label(label, ignore_labels):
    if isinstance(ignore_labels, int):
        ignore_labels = [ignore_labels]
    mask = label >= 0
    for ignore_label in ignore_labels:
        mask = np.logical_and(mask, label != ignore_label)
    label = label * np.int32(mask)
    return label


def label_db_transform(label, with_channel=False,
        ignore_labels=None):
    #label = np.float32(label)/class_num
    if ignore_labels is not None:
        label=mask_label(label, ignore_labels)


    label = label[NEW_X, NEW_X, :, :] if with_channel \
            else label[NEW_X, :, :]

    return label # (1, h, w)

