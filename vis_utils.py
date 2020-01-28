import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb
b = pdb.set_trace

def dump_prob2image(filename, array : np.array):
    """
        dump probility map to image when
        array: [x, height, width] (x = 1, 3, 4)
    """
    class_num = array.shape[0]
    assert class_num <= 4
    if class_num == 2:
        raise ValueError('not implement')

    array = np.transpose(np.uint8(array * 255), (1, 2, 0))
    cv2.imwrite(filename + '.png', array)

def load_image2prob(filename):
    if not filename.endswith('.png'):
        filename = filename + '.png'
    array = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    array = np.transpose(array, (2, 0, 1)) / 255
    return array

def mask2box(mask):
    """
        t, l, b, r
    """
    y, x = np.where(mask > 0)
    return [np.min(y), np.min(x), np.max(y), np.max(x)]

def dilate_mask(mask, kernel=20):
    kernel = np.ones((kernel, kernel), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

def pack_argument(args, arg_names):
    """
    args: object of all arguments
    arg_names: list of string name for needed arguments
    """
    kwargs = {}
    for arg_name in arg_names:
        cur_args = getattr(args, arg_name) if hasattr(args, arg_name) else None
        if cur_args:
            kwargs[arg_name] = cur_args

    return kwargs

def split_list(seq, part):
    """split a list to sub lists
    """
    size = len(seq) / part + 1
    size = int(size)

    return [seq[i:i+size] for i  in range(0, len(seq), size)]


def mkdir_if_need(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def mkdir_if_exists(path, image_name):
    target_path = path + os.path.dirname(image_name)
    if not os.path.exists(target_path):
        os.makedirs(target_path)

def save_image_w_pallete(segment, file_name):
    import PIL.Image as Image
    pallete = get_pallete(256)

    segmentation_result = np.uint8(segment)
    segmentation_result = Image.fromarray(segmentation_result)
    segmentation_result.putpalette(pallete)
    segmentation_result.save(file_name)


def get_pallete(num_cls):
    """
    this function is to get the colormap for visualizing
    the segmentation mask
    :param num_cls: the number of visulized class
    :return: the pallete
    """
    n = num_cls
    pallete = [0]*(n*3)
    for j in range(0,n):
        lab = j
        pallete[j*3+0] = 0
        pallete[j*3+1] = 0
        pallete[j*3+2] = 0
        i = 0
        while (lab > 0):
                pallete[j*3+0] |= (((lab >> 0) & 1) << (7-i))
                pallete[j*3+1] |= (((lab >> 1) & 1) << (7-i))
                pallete[j*3+2] |= (((lab >> 2) & 1) << (7-i))
                i = i + 1
                lab >>= 3
    return pallete


def color2label(label_color, color_map):
    """
    Convert color image to semantic id based on color_map
    """
    # default bkg 255
    label_color = np.int32(label_color)
    height, width = label_color.shape[0:2]
    label = label_color[:, :, 0] * (255 ** 2) + \
            label_color[:, :, 1] * 255 + \
            label_color[:, :, 2]

    label_id = np.unique(label)
    for rgb, i in color_map.items():
        cur_num = rgb[0] * (255 ** 2) + rgb[1] * 255 + rgb[2]
        if cur_num in label_id:
            mask = (label - cur_num) != 0
            label = label * mask  + i * (1 - mask)

    return label

def show_grey(image):
    image = image.squeeze()
    assert len(image.shape) == 2
    plt.imshow(image, cmap='gray')


def flow2color(flow):
    assert flow.shape[2] == 2
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3),
        dtype=np.float32)
    hsv[...,1] = 255
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(np.uint8(hsv),cv2.COLOR_HSV2BGR)
    return hsv, rgb


def colorEncode(labelmap, colors, mode='RGB'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in np.unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb

def drawBoundingbox(image, boxes, colors=None):
    """

    """
    if colors is None:
        colors = [[255, 255, 0]] * len(boxes)

    for color, box in zip(colors, boxes):
        box = box.astype(np.uint32)
        t, l, b, r = box[0], box[1], box[2], box[3]
        cv2.rectangle(image, (l, t), (r, b), color, 2)

    return image


def resize2maxsize(image, size=720, interpolation=None):
    """
        Constraint the maximum length of an image
    """
    if interpolation is None:
        interpolation = cv2.INTER_CUBIC

    height, width = image.shape[:2]
    scale = max(height, width) / size
    image_out = image.copy()
    if scale > 1:
        height, width = np.uint32(np.array(image.shape[:2]) / scale)
        image_out = cv2.resize(image_out, (width, height),
                               interpolation=interpolation)

    return image_out

def resize2range(image, max_size=720, min_size=480,
                 interpolation=None):
    """
        Constraint the maximum length of an image and min size of an image
        if conf
    """
    if interpolation is None:
        interpolation = cv2.INTER_LINEAR

    height, width = image.shape[:2]
    scale_to_max = max_size / max(height, width)
    scale_to_min = min(min_size / min(height, width),
                       max_size / max(height, width))

    image_out = image.copy()
    if scale_to_max < 1:
        height, width = np.uint32(np.array(image.shape[:2]) * scale_to_max)
        image_out = cv2.resize(image_out, (width, height),
                               interpolation=interpolation)
        return image_out

    if scale_to_min > 1:
        height, width = np.uint32(np.array(image.shape[:2]) * scale_to_min)
        image_out = cv2.resize(image_out, (width, height),
                               interpolation=interpolation)
        return image_out

    return image_out


def resize2size(image, size=720, interpolation=None):
    """
        Constraint the maximum size to be size
        make the longer edge equal to size
    """
    if interpolation is None:
        interpolation = cv2.INTER_CUBIC

    height, width = image.shape[:2]
    scale = max(height, width) / size

    image_out = image.copy()
    height, width = np.uint32(np.array(image.shape[:2]) / scale)
    image_out = cv2.resize(image_out, (width, height),
                           interpolation=interpolation)

    return image_out


def resize2maxshape(image, shape, interpolation=None):
    """
        shape is the target video shape
        resize an image to target shape by padding zeros
            when ratio is not match
    """
    def get_start_end(scale_id, height_new, width_new):
        if scale_id == 0:
            s_v, e_v = 0, height_new
            s_h = int((shape[1] - width_new) / 2)
            e_h = s_h + width_new
        else:
            s_v = int((shape[0] - height_new) / 2)
            e_v = s_v + height_new
            s_h, e_h = 0, width_new
        return s_v, e_v, s_h, e_h

    if interpolation is None:
        interpolation = cv2.INTER_CUBIC

    image_shape = shape if image.ndim == 2 else shape + [image.shape[-1]]
    image_out = np.zeros(image_shape)
    height, width = image.shape[:2]
    scale_rate = np.array([shape[0] / height, shape[1] / width])
    scale_id = np.argmin(scale_rate)
    scale = scale_rate[scale_id]
    image = cv2.resize(image,
                       (int(width * scale), int(height * scale)),
                       interpolation=interpolation)
    height_new, width_new = image.shape[:2]
    s_v, e_v, s_h, e_h = get_start_end(scale_id, height_new, width_new)
    image_out[s_v:e_v, s_h:e_h] = image

    return image_out



def show_flow(flow):
    hsv, rgb = flow2color(flow)
    plt.imshow(np.uint8(rgb))


def show_depth(depth):
    plt.imshow(depth, cmap=plt.get_cmap('jet'))


def show_grey_rev(image):
    assert len(image.shape.squeeze()) == 2
    plt.imshow(1.0 - image, cmap='gray')


vis_func = {'gray': show_grey,
            'rev_gray': show_grey,
            'flow': show_flow,
            'depth': show_depth,
            'color': plt.imshow,
            'cv_color': cv2.imshow}

def plot_images(images, layout=[2,2],
                fig_size=10, attr=None,
                save_fig=False, is_close=False,
                fig=None, fig_name='tmp.jpg'):

    import matplotlib.pylab as pylab

    is_show = True if fig is None else False
    if fig is None:
        fig = plt.figure(figsize=(10,5))

    pylab.rcParams['figure.figsize'] = fig_size, fig_size/2
    Keys = images.keys()
    attr_all = {}
    for iimg, name in enumerate(Keys):
        # not support the one dim data
        assert len(images[name].shape) >= 2

        if len(images[name].shape) == 2:
            attr_all[name] = 'depth'
        else:
            if images[name].shape[2] == 2:
                attr_all[name] = 'flow'
            else:
                attr_all[name] = 'color'

    if attr:
        attr_all.update(attr)

    for iimg, name in enumerate(Keys):
        # print(name)
        s = plt.subplot(layout[0], layout[1], iimg+1)

        vis_func[attr_all[name]](images[name])

        s.set_xticklabels([])
        s.set_yticklabels([])
        s.set_title(name)
        s.yaxis.set_ticks_position('none')
        s.xaxis.set_ticks_position('none')

    plt.tight_layout()

    if save_fig:
        pylab.savefig(fig_name)
    else:
        if is_show:
            plt.show()
        else:
            fig.canvas.draw()


def dump_to_npy(arrays, file_path=None):
    """
       dump set of images to array for local visualization
       arrays: the input arrays
       file_path: saving path
    """
    assert isinstance(arrays, dict)
    for k, v in arrays.items():
        np.save(os.path.join(file_path, k + '.npy'), v)


def crop(image, box):
    """
    box: t, l, b, r
    """
    t, l, b, r = box
    return image[t:b, l:r]


def padding_image(image_in,
                  image_size,
                  crop=None,
                  interpolation=cv2.INTER_NEAREST,
                  pad_val=0.):

    """Pad image to target image_size based on a given crop
    """
    assert isinstance(pad_val, float) | isinstance(pad_val, list)

    if image_size[0] <= image_in.shape[0] and \
            image_size[1] <= image_in.shape[1]:
        return image_in

    image = image_in.copy()
    in_dim = np.ndim(image)
    if in_dim == 2:
        image = image[:, :, None]

    if isinstance(pad_val, float):
        pad_val = [pad_val] * image.shape[-1]
    assert len(pad_val) == image.shape[-1]

    dim = image.shape[2]
    image_pad = np.ones(image_size + [dim], dtype=image_in.dtype) * \
        np.array(pad_val)

    if not (crop is None):
        h, w = image_size
        crop_cur = np.uint32([crop[0] * h, crop[1] * w,
                              crop[2] * h, crop[3] * w])
        image = cv2.resize(
            image, (crop_cur[3] - crop_cur[1], crop_cur[2] - crop_cur[0]),
            interpolation=interpolation)

    else:
        h, w = image_in.shape[:2]
        # default crop is padding right and down
        crop_cur = [0, 0, h, w]
    image_pad[crop_cur[0]:crop_cur[2], crop_cur[1]:crop_cur[3], :] = image

    if in_dim == 2:
        image_pad = np.squeeze(image_pad)

    return image_pad


def one_hot(label_map, class_num):
    shape = np.array(label_map.shape)
    length = np.prod(shape)
    label_one_hot = np.zeros((length, class_num))
    label_flat = label_map.flatten()
    label_one_hot[range(length), label_flat] = 1
    label_one_hot = label_one_hot.reshape(shape.tolist() + [class_num])

    return label_one_hot


def prob2label(label_prob):
    """Convert probability to a descrete label map
    """
    assert label_prob.ndim == 3
    return np.argmax(label_prob, axis=2)


def prob2color(label_prob, color_map, bkg_color=[0,0,0]):
    height, width, dim = label_prob.shape

    color_map_mat = np.matrix([bkg_color] + color_map)
    label_prob_mat = np.matrix(label_prob.reshape((height * width, dim)))
    label_color = np.array(label_prob_mat * color_map_mat)
    label_color = label_color.reshape((height, width, -1))

    return np.uint8(label_color)


def label2color(label, color_map, bkg_color=[0, 0, 0]):
    height, width = label.shape[0:2]
    class_num = len(color_map) + 1
    label_one_hot = one_hot(label, class_num)
    label_color = prob2color(label_one_hot, color_map, bkg_color)

    return label_color


def video_to_frames(in_path, out_path, max_frame=100000):
    """separate video to frames
    """
    print("saving videos to frames at {}".format(out_path))
    cap = cv2.VideoCapture(in_path)
    frame_id = 0
    mkdir_if_need(out_path)

    cv2.namedWindow("video")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        filename = out_path + 'frame {}.jpg'.format(str(frame_id))
        print(filename)
        cv2.imshow('video',frame)
        cv2.imwrite(filename, frame)
        frame_id += 1
        if frame_id > max_frame:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("finished")


def frame_to_video(image_path,
                   label_path,
                   frame_list,
                   label_ext='',
                   is_color=False,
                   color_map=None,
                   sz=None,
                   fps=10,
                   alpha=0.5,
                   video_name='video.avi'):
    """Combine frames to video
    """
    import sys

    if sz is None:
        label = cv2.imread("%s/%s.png" % (label_path, frame_list[0]))
        sz = label.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    video = cv2.VideoWriter(video_name, fourcc, fps, (sz[1], sz[0]))
    for i, image_name in enumerate(frame_list):
        sys.stdout.write('\r>>process %04d / %04d' % (i, len(frame_list)))
        sys.stdout.flush()

        image = cv2.resize(cv2.imread("%s%s.jpg" % (image_path, image_name),
            cv2.IMREAD_UNCHANGED),
            (sz[1], sz[0]))
        label_name = image_name + label_ext
        label = cv2.resize(cv2.imread("%s%s.png" % (label_path, label_name),
            cv2.IMREAD_UNCHANGED),
            (sz[1], sz[0]), interpolation=cv2.INTER_NEAREST)

        if not is_color:
            assert color_map is not None
            bkg = [255, 255, 255]
            label[label > len(color_map)] = 0
            label = label2color(label, color_map, bkg)
            label = label[:, :, ::-1]

        frame = np.uint8(image * alpha + label * (1 - alpha))
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()


def test_one_hot():
    label = np.array([[1, 2], [3, 4]])
    label_one_hot = one_hot(label, 5)
    print(label_one_hot)


def test_resize2range():
    test = np.ones([100, 200])
    test2 = resize2range(test, 200, 50)
    print(test2.shape)

def test_prob2image():
    test = np.random.random_sample((3, 10, 10))
    dump_prob2image('test', test)
    res = load_image2prob('test')

    np.testing.assert_allclose(test, res, rtol=0.5, atol=1e-02)


if __name__ == '__main__':
    # test_one_hot()
    # test_resize2range()
    test_prob2image()
