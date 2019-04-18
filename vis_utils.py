
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb


def split_list(seq, part):
    """split a list to sub lists
    """
    size = len(seq) / part + 1
    size = int(size)

    return [seq[i:i+size] for i  in range(0, len(seq), size)]


def mkdir_if_need(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)



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

    if sz is None:
        label = cv2.imread("%s%s.png" % (label_path, frame_list[0]))
        sz = label.shape

    fourcc = cv2.cv.CV_FOURCC(*'DIV3')
    video = cv2.VideoWriter(video_name, fourcc, fps, (sz[1], sz[0]))
    for i, image_name in enumerate(frame_list):
        print "compress %04d" % i
        image = cv2.resize(cv2.imread("%s%s.jpg" % (image_path, image_name),
            cv2.IMREAD_UNCHANGED),
            (sz[1], sz[0]))
        label_name = image_name + label_ext
        label = cv2.resize(cv2.imread("%s%s.png" % (label_path, label_name),
            cv2.IMREAD_UNCHANGED),
            (sz[1], sz[0]), interpolation=cv2.INTER_NEAREST)

        if not is_color:
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


if __name__ == '__main__':
    test_one_hot()
