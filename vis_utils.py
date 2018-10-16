
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


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

    return [seq[i:i+size] for i  in range(0, len(seq), size)]


def mkdir_if_need(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)



def plot_images(images, layout=[2,2], fig_size=10, attr=None, save_fig=False, is_close=False, fig=None, fig_name='tmp.jpg'):

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
