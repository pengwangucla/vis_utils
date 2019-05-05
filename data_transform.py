#############
# data transforms for neural network transform, from a 2D image to 4D blob
#############
import numpy as np
import utils_3d as uts_3d

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


def label_transform(label, label_mapping=None):
    """
        Transform gt label to the training and evaluation label id
    """
    if label_mapping is not None:
        label = np.uint8(label)
        label = np.float32(label_mapping[label])

    return label_db_transform(label)


def pose_transform(pose,
                   mean_pose=None,
                   scale=None,
                   to_quater=False):
    """
    Transform pose to network inputs
    Inputs:
        to_quater: to the quaternion representation
    """


    if mean_pose is None:
        mean_pose = np.zeros(3, dtype=np.float32)
    if scale is None:
        scale = 1.0

    pose[:3] = (pose[:3] - mean_pose)/scale
    if to_quater:
        pose = np.concatenate([pose[:3],
            uts_3d.euler_angles_to_quaternions(pose[3:])])

    return pose[NEW_X, :]


def point_transform(points):
    """ Transform 3d points
    """
    points = np.transpose(points, [1, 0])
    return np.expand_dims(points, axis=0)


def flow_transform(flow):
    """ Transform optical flow
    """

    flow = np.transpose(flow, [2, 0, 1])
    return np.expand_dims(flow, axis=0) #(1, 2, h, w)

