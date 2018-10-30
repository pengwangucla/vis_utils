"""
Evaluation metrics.
"""

import numpy as np
import utils_3d as uts_3d
import enum
from collections import OrderedDict
import pdb


class InputType(enum.IntEnum):
    MXNET = 1
    NUMPY = 2

    @classmethod
    def has_value(cls, value):
        return any(value == item.value for item in cls)


class FlowMetric(object):
    """EPE metric, including mean-EPE
    """
    def __init__(self, output_names=None, label_names=None, mask=None):
        """Initializer.
        """
        # super(FlowMetric, self).__init__('FlowMetric', output_names, label_names)
        self._names = ['mean_epe']
        self._offset = 0

    def reset(self):
        """Reset metrics.
        """
        self._offset = 0
        self._pixel_num = 0

    def update(self, labels, preds):
        """Update metrics. we assume flow is in shape [batch, 2, height, width]
        """
        for pred, label in zip(preds, labels):
            label = label.asnumpy()
            pred = pred.asnumpy()
            mask = np.float32(np.logical_and(label[:, 0, :, :] != 0,
                                             label[:, 1, :, :] != 0))
            diff = label - pred
            diff = np.sum(np.sqrt(np.sum(diff * diff, axis=1) * mask))

            self._offset += diff
            self._pixel_num += np.sum(mask)

    def get(self):
        """Get current state of metrics.
        """
        mean_offset = self._offset / self._pixel_num
        values = [mean_offset]
        return (self._names, values)


class PoseMetric(object):
    """Segmentation metric, including pixel-acc, mean-acc, mean-iou
    """
    def __init__(self, output_names=None, label_names=None, is_euler=False,
            trans_idx=None, rot_idx=None, data_type=InputType.MXNET):
        """ Initializer.
        """
        super(PoseMetric, self).__init__('PoseMetric', output_names, label_names)
        self._names = ['mean_xyz', 'median_xyz', 'mean_theta', 'median_theta']
        self._offset = []
        self._theta = []
        self._is_euler = is_euler
        self._trans_idx = [0, 1, 2] if trans_idx is None else trans_idx
        if is_euler:
            self._rot_idx = [3, 4, 5] if rot_idx is None else rot_idx
        else:
            self._rot_idx = [3, 4, 5, 6] if rot_idx is None else rot_idx
        self.max_val = 1000.
        assert InputType.has_value(data_type)
        self.data_type = data_type


    def reset(self):
        """Reset metrics.
        """
        self._offset = []
        self._theta = []
        self._ind = []


    def convert2np(self, pred, label):
        if self.data_type == InputType.MXNET:
            label = label.asnumpy()
            pred = pred.asnumpy()

        return pred, label


    def update(self, labels, preds):
        """Update metrics.
        """
        for pred, label in zip(preds, labels):
            pred, label = self.convert2np(pred, label)

            pose_x = label[:, self._trans_idx]
            pred_x = pred[:, self._trans_idx]
            valid_ind = np.sum(np.abs(label), axis=1) > 0

            if self._is_euler:
                pose_q = uts_3d.euler_angles_to_quaternions(
                        label[:, self._rot_idx])
                pred_q = uts_3d.euler_angles_to_quaternions(
                        pred[:, self._rot_idx])
            else:
                pose_q = label[:, self._rot_idx]
                pred_q = pred[:, self._rot_idx]

            q1 = pose_q / np.linalg.norm(pose_q, axis=1)[:, None]
            q2 = pred_q / np.linalg.norm(pred_q, axis=1)[:, None]
            diff = abs(np.ravel(1 - np.sum(np.square((q1 - q2)) / 2, axis=1)))
            self._theta.append(2 * np.arccos(diff) * 180 / np.pi)
            self._offset.append(np.ravel(np.linalg.norm(pose_x - pred_x, axis=1)))
            self._ind.append(valid_ind)


    def get(self):
        """Get current state of metrics.
        """
        if len(self._offset) == 0 or len(self._theta) == 0:
            values = [self.max_val for i in self._names]
        else:
            all_offset = np.concatenate(self._offset)
            all_theta = np.concatenate(self._theta)
            all_ind = np.concatenate(self._ind)
            median_offset = np.median(all_offset[all_ind])
            median_theta = np.median(all_theta)
            mean_offset = np.mean(all_offset)
            mean_theta = np.mean(all_theta)
            values = [mean_offset, median_offset, mean_theta, median_theta]

        return (self._names, values)



class MultiTagMetric(object):
    """Use pascal mean ap metric for multi-class evaluation with score
    """
    def __init__(self, output_names=None, label_names=None,
                 ignore_label=None):
        """Initializer. we may evaluate multiple outputs
        Inputs:
            ignore_label can be a list or a integer
        """
        self._names = ['mean_ap']
        # whether ignore certain class
        if ignore_label is None:
            self._ignore_labels = []
        self.output_names = output_names
        self.label_names = label_names

        if isinstance(ignore_label, list):
            self._ignore_labels = ignore_label
        elif isinstance(ignore_label, int):
            self._ignore_labels = [ignore_label]


        self._preds = [None for i in output_names]
        self._gts = [None for i in label_names]
        self._num_inst = None
        self._nclass = 0

    def reset(self):
        """Reset metrics.
        """
        self._preds = [None for i in self.output_names]
        self._gts = [None for i in self.label_names]
        self._num_inst = None
        self._nclass = 0

    def update(self, labels, preds):
        """Update metrics.
        """
        for out_id, (pred, label) in enumerate(zip(preds, labels)):
            if self._gts[out_id] is None:
                self._nclass = pred.shape[1]
                self._gts[out_id] = label
                self._preds[out_id] = pred
                self._num_inst[out_id] = np.sum(label, axis=0)
                continue

            self._gts[out_id] = np.concatenate((self._gts[out_id], label),
                    axis=0)
            self._preds[out_id] = np.concatenate((self._preds[out_id], pred),
                    axis=0)
            self._num_inst[out_id] += np.sum(label, axis=0)


    def get(self):
        """Get current state of metrics.
        """
        full_res = np.zeros([len(self._gts), len(self._names)])
        for out_id, (preds, gts) in zip(self._gts, self._preds):
            output = self.evaluate(preds, gts)
            full_res[out_id, :] = np.array(output.values)

        return (self._names, np.mean(full_res, axis=0))


    def evaluate(self, preds, gts):
        '''
        Evaluation of image classification task
        Run per image evaluation on all images in the set
        :return: map
        '''
        voc_ap = np.zeros(self.num_cats)
        record = {}
        record['cat_ap'] = []
        for eval_order in range(self._nclass):
            if eval_order in self._ignore_labels:
                continue
            voc_eval = self.evalcls(preds[eval_order, :], gts[eval_order, :])
            voc_ap[eval_order] = voc_eval['ap']

        # average ap
        res = OrderedDict([])
        res['map'] = voc_ap.sum()/float(len(record['cat_ap']))
        return res


    def VOCap(self, rec, prec):
        ''' average percision calculations
        '''
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def VOCevalcls(self, pred, gt):
        '''Evaluate per category AP and average over number of category to get meanAP
           Follow the evaluation in VOC2010 Toolbox
        '''
        si = np.argsort(-pred)

        tp = gt[si]==1
        fp = gt[si]==0
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)

        rec = tp/float((gt>0).sum()) if (gt>0).sum() else tp*0
        prec = tp/np.maximum(fp+tp, np.finfo(np.float64).eps) # if (fp+tp) else 0 # aviod devide by 0
        ap = self.VOCap(rec, prec)

        return {'ap': ap, 'rec': rec, 'prec': prec}



class SegMetric(object):
    """Segmentation metric, including pixel-acc, mean-acc, mean-iou
    """
    def __init__(self, output_names=None, label_names=None,
            use_ignore=True, ignore_label=255):
        """Initializer.
        Inputs:
            ignore_label can be a list or a integer
        """
        super(SegMetric, self).__init__('SegMetric', output_names, label_names)
        self._names = ['pixel-acc', 'mean-acc', 'mean-iou']
        self._use_ignore = use_ignore
        if isinstance(ignore_label, list):
            self._ignore_labels = ignore_label
        elif isinstance(ignore_label, int):
            self._ignore_labels = [ignore_label]

        self._tp = None
        self._fp = None
        self._fn = None
        self._num_inst = None
        self._ious = None
        self._nclass = 0

    def reset(self):
        """Reset metrics.
        """
        self._tp = None
        self._fp = None
        self._fn = None
        self._num_inst = None
        self._ious = None
        self._nclass = 0

    def update(self, labels, preds):
        """Update metrics.
        """
        for pred, label in zip(preds, labels):
            if self._nclass == 0:
                self._nclass = pred.shape[1]
                self._tp = np.zeros(self._nclass)
                self._fp = np.zeros(self._nclass)
                self._fn = np.zeros(self._nclass)
                self._num_inst = np.zeros(self._nclass)

            label = label.ravel()
            pred = pred.argmax(1).ravel()

            if self._use_ignore:
                mask = label >= 0
                for ignore_label in self._ignore_labels:
                    mask = np.logical_and(mask, label != ignore_label)

                label = label[mask]
                pred = pred[mask]

            for i in xrange(self._nclass):
                self._tp[i] += ((label == i) & (pred == i)).sum()
                self._fp[i] += ((label != i) & (pred == i)).sum()
                self._fn[i] += ((label == i) & (pred != i)).sum()
                self._num_inst[i] += (label == i).sum()

    def get(self):
        """Get current state of metrics.
        """
        pixel_acc = self._tp.sum() / self._num_inst.sum()
        accs = np.divide(self._tp, self._num_inst,
               out=np.full_like(self._tp, np.nan), where=self._num_inst != 0)
        self._ious = np.divide(self._tp, self._tp + self._fp + self._fn,
               out=np.full_like(self._tp, np.nan), where=self._num_inst != 0)
        values = [pixel_acc,
                  accs[np.logical_not(np.isnan(accs))].mean(),
                  self._ious[np.logical_not(np.isnan(self._ious))].mean()]

        return (self._names, values)

    def get_ious(self):
        """Get current ious.
        """
        return self._ious

    def get_pixel_num(self):
        """Get current pixel num.
        """
        return self._num_inst




if __name__ == '__main__':
    ignore_label = 4
    seg_metric = SegMetric(ignore_label=ignore_label)
    seg = np.array([1, 2, 3, 1, 2, 3, 4])[None, :]
    seg = np.one_hot(np.array(seg), 5)
    seg = np.transpose(seg, axes=(0, 2, 1))
    seg_gt = np.array([1, 2, 4, 1, 2, 3, 4])[None, :]
    seg_metric.update([np.array(seg_gt)], [seg])
    print seg.shape, seg_gt.shape
    print seg_metric.get()

    pose_metric = PoseMetric()
    pose = np.array([1, 2, 3, 1, 2, 3., 4])[None, :]
    pose_gt = np.array([1, 2, 4, 1, 2, 3., 4])[None, :]

    pose_metric.update([np.array(pose_gt)], [np.array(pose)])
    print pose_metric.get()

    flow_metric = FlowMetric()
    flow = np.array([1, 2, 3, 1, 2, 3., 4, 5])
    flow_gt = np.array([1, 2, 4, 1, 2, 3., 4, 6])
    flow = flow.reshape((1, 2, 2, 2))
    flow_gt = flow_gt.reshape((1, 2, 2, 2))
    flow_metric.update([np.array(flow_gt)], [np.array(flow)])
    print flow_metric.get()


