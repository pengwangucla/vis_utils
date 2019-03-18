
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb


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
    for j in xrange(0,n):
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

def split_list(seq, part):
    """split a list to sub lists
    """
    size = len(seq) / part + 1
    size = int(size)

    return [seq[i:i+size] for i  in range(0, len(seq), size)]


def mkdir_if_need(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)



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
