import os
import random

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter


class DataGenerator:
    def __init__(self, data_dir, target_size=256):
        self.data_dir = data_dir
        self.target_size = target_size

        os.makedirs(os.path.join(self.data_dir, 'images/train'), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'images/test'), exist_ok=True)

    def process_300w(self, folder_name, image_name, label_name):
        image_path = os.path.join(self.data_dir, folder_name, image_name)
        label_path = os.path.join(self.data_dir, folder_name, label_name)

        with open(label_path, 'r') as ff:
            anno = ff.readlines()[20:-1]
            anno = [x.strip().split() for x in anno]
            anno = [[int(float(x[0])), int(float(x[1]))] for x in anno]
            image = cv2.imread(image_path)
            image_height, image_width, _ = image.shape
            anno_x = [x[0] for x in anno]
            anno_y = [x[1] for x in anno]
            bbox_xmin = min(anno_x)
            bbox_ymin = min(anno_y)
            bbox_xmax = max(anno_x)
            bbox_ymax = max(anno_y)
            bbox_width = bbox_xmax - bbox_xmin
            bbox_height = bbox_ymax - bbox_ymin
            scale = 1.1
            bbox_xmin -= int((scale - 1) / 2 * bbox_width)
            bbox_ymin -= int((scale - 1) / 2 * bbox_height)
            bbox_width *= scale
            bbox_height *= scale
            bbox_width = int(bbox_width)
            bbox_height = int(bbox_height)
            bbox_xmin = max(bbox_xmin, 0)
            bbox_ymin = max(bbox_ymin, 0)
            bbox_width = min(bbox_width, image_width - bbox_xmin - 1)
            bbox_height = min(bbox_height, image_height - bbox_ymin - 1)
            anno = [[(x - bbox_xmin) / bbox_width, (y - bbox_ymin) / bbox_height] for x, y in anno]

            bbox_xmax = bbox_xmin + bbox_width
            bbox_ymax = bbox_ymin + bbox_height
            image_crop = image[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax, :]
            image_crop = cv2.resize(image_crop, (self.target_size, self.target_size))
            return image_crop, anno

    def process_dataset(self, folders, subset):
        annotations = {}
        for folder in folders:
            img_path = os.path.join(self.data_dir, folder)
            files = sorted(os.listdir(img_path))
            images = [x for x in files if '.pts' not in x]
            labels = [x for x in files if '.pts' in x]
            assert len(images) == len(labels)

            for image, label in zip(images, labels):
                print(f'{image}---{label}')
                cropped_image, anno = self.process_300w(folder, image, label)
                image_crop_name = folder.replace('/', '_') + '_' + image
                image_crop_name = os.path.join(self.data_dir, 'images', subset, image_crop_name)
                cv2.imwrite(image_crop_name, cropped_image)
                annotations[image_crop_name] = anno

            with open(os.path.join(self.data_dir, f'images/{subset}.txt'), 'w') as f:
                for image_crop_name, anno in annotations.items():
                    f.write(image_crop_name + ' ')
                    for x, y in anno:
                        f.write(str(x) + ' ' + str(y) + ' ')
                    f.write('\n')

    def split_data(self):
        with open(os.path.join(self.data_dir, 'images/test.txt'), 'r') as f:
            annos = f.readlines()
        with open(os.path.join(self.data_dir, 'images/test_common.txt'), 'w') as f:
            for anno in annos:
                if not 'ibug' in anno:
                    f.write(anno)
        with open(os.path.join(self.data_dir, 'images/test_challenge.txt'), 'w') as f:
            for anno in annos:
                if 'ibug' in anno:
                    f.write(anno)

    def get_indices(self, file):
        with open(file, 'r') as f:
            annotations = f.readlines()
        annotations = [x.strip().split()[1:] for x in annotations]
        annotations = np.array([[float(x) for x in anno] for anno in annotations])
        indices = np.mean(annotations, axis=0)
        indices = indices.tolist()
        indices = [str(x) for x in indices]

        with open(os.path.join(self.data_dir, 'images/indices.txt'), 'w') as f:
            f.write(' '.join(indices))

    def run(self):
        self.process_dataset([], 'train')
        self.process_dataset(['helen/testset', 'lfpw/testset', 'ibug'], 'test')
        self.split_data()
        self.get_indices(os.path.join(self.data_dir, 'images/train.txt'))


def compute_indices(mean_face_file, params):
    with open(mean_face_file) as f:
        mean_face = f.readlines()[0]

    mean_face = mean_face.strip().split()
    mean_face = [float(x) for x in mean_face]
    mean_face = np.array(mean_face).reshape(-1, 2)
    # each landmark predicts num_nb neighbors
    mean_indices = []
    for i in range(mean_face.shape[0]):
        pt = mean_face[i, :]
        dists = np.sum(np.power(pt - mean_face, 2), axis=1)
        indices = np.argsort(dists)
        mean_indices.append(indices[1:1 + params['num_nb']])

    # each landmark predicted by X neighbors, X varies
    mean_face_indices_reversed = {}
    for i in range(mean_face.shape[0]):
        mean_face_indices_reversed[i] = [[], []]
    for i in range(mean_face.shape[0]):
        for j in range(params['num_nb']):
            mean_face_indices_reversed[mean_indices[i][j]][0].append(i)
            mean_face_indices_reversed[mean_indices[i][j]][1].append(j)

    max_len = 0
    for i in range(mean_face.shape[0]):
        if len(mean_face_indices_reversed[i][0]) > max_len:
            max_len = len(mean_face_indices_reversed[i][0])

    # tricks, make them have equal length for efficient computation
    for i in range(mean_face.shape[0]):
        mean_face_indices_reversed[i][0] += mean_face_indices_reversed[i][0] * 10
        mean_face_indices_reversed[i][1] += mean_face_indices_reversed[i][1] * 10
        mean_face_indices_reversed[i][0] = mean_face_indices_reversed[i][0][:max_len]
        mean_face_indices_reversed[i][1] = mean_face_indices_reversed[i][1][:max_len]

    # make the indices 1-dim
    reverse_index1 = []
    reverse_index2 = []
    for i in range(mean_face.shape[0]):
        reverse_index1 += mean_face_indices_reversed[i][0]
        reverse_index2 += mean_face_indices_reversed[i][1]
    return mean_indices, reverse_index1, reverse_index2, max_len


class ComputeLoss:
    def __init__(self, params):
        super().__init__()
        self.cls = params['crit_cls']
        self.reg = params['crit_reg']
        self.num_neighbor = params['num_nb']
        self.criterion_reg = torch.nn.L1Loss()
        self.criterion_cls = torch.nn.MSELoss()

    def __call__(self, outputs, targets):
        device = outputs[0].device
        b, c, h, w = outputs[0].size()

        score = outputs[0]
        offset_x = outputs[1].view(b * c, -1)
        offset_y = outputs[2].view(b * c, -1)
        neighbor_x = outputs[3].view(b * self.num_neighbor * c, -1)
        neighbor_y = outputs[4].view(b * self.num_neighbor * c, -1)

        target_score = targets[0].to(device).view(b * c, -1)
        target_offset_x = targets[1].to(device).view(b * c, -1)
        target_offset_y = targets[2].to(device).view(b * c, -1)
        target_neighbor_x = targets[3].to(device).view(b * self.num_neighbor * c, -1)
        target_neighbor_y = targets[4].to(device).view(b * self.num_neighbor * c, -1)

        target_max_index = torch.argmax(target_score, 1).view(-1, 1)
        target_max_index_neighbor = target_max_index.repeat(1, self.num_neighbor).view(-1, 1)

        offset_x_select = torch.gather(offset_x, 1, target_max_index)
        offset_y_select = torch.gather(offset_y, 1, target_max_index)
        neighbor_x_select = torch.gather(neighbor_x, 1, target_max_index_neighbor)
        neighbor_y_select = torch.gather(neighbor_y, 1, target_max_index_neighbor)

        target_offset_x_select = torch.gather(target_offset_x, 1, target_max_index)
        target_offset_y_select = torch.gather(target_offset_y, 1, target_max_index)
        target_neighbor_x_select = torch.gather(target_neighbor_x, 1, target_max_index_neighbor)
        target_neighbor_y_select = torch.gather(target_neighbor_y, 1, target_max_index_neighbor)

        loss_cls = self.criterion_cls(score, target_score.view(b, c, h, w))
        loss_offset_x = self.criterion_reg(offset_x_select, target_offset_x_select)
        loss_offset_y = self.criterion_reg(offset_y_select, target_offset_y_select)
        loss_neighbor_x = self.criterion_reg(neighbor_x_select, target_neighbor_x_select)
        loss_neighbor_y = self.criterion_reg(neighbor_y_select, target_neighbor_y_select)

        loss_cls = self.cls * loss_cls
        loss_reg = self.reg * (loss_offset_x + loss_offset_y + loss_neighbor_x + loss_neighbor_y)
        return loss_cls + loss_reg


def compute_nme(output, target, norm):
    output = output.reshape((-1, 2))
    target = target.reshape((-1, 2))
    return np.mean(np.linalg.norm(output - target, axis=1)) / norm


def compute_fr_and_auc(nme, thresh=0.1, step=0.0001):
    from scipy.integrate import simps
    num_data = len(nme)
    xs = np.arange(0, thresh + step, step)
    ys = np.array([np.count_nonzero(nme <= x) for x in xs]) / float(num_data)
    fr = 1.0 - ys[-1]
    auc = simps(ys, x=xs) / thresh
    return fr, auc


def distance2box(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    outputs = []
    for i in range(0, distance.shape[1], 2):
        p_x = points[:, i % 2] + distance[:, i]
        p_y = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            p_x = p_x.clamp(min=0, max=max_shape[1])
            p_y = p_y.clamp(min=0, max=max_shape[0])
        outputs.append(p_x)
        outputs.append(p_y)
    return np.stack(outputs, axis=-1)


class FaceDetector:
    def __init__(self, onnx_path=None, session=None):
        from onnxruntime import InferenceSession
        self.session = session

        self.batched = False
        if self.session is None:
            assert onnx_path is not None
            assert os.path.exists(onnx_path)
            self.session = InferenceSession(onnx_path,
                                            providers=['CUDAExecutionProvider'])
        self.nms_thresh = 0.4
        self.center_cache = {}
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
        input_name = input_cfg.name
        outputs = self.session.get_outputs()
        if len(outputs[0].shape) == 3:
            self.batched = True
        output_names = []
        for output in outputs:
            output_names.append(output.name)
        self.input_name = input_name
        self.output_names = output_names
        self.use_kps = False
        self._num_anchors = 1
        if len(outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def forward(self, x, score_thresh):
        scores_list = []
        bboxes_list = []
        points_list = []
        input_size = tuple(x.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(x,
                                     1.0 / 128,
                                     input_size,
                                     (127.5, 127.5, 127.5), swapRB=True)
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            if self.batched:
                scores = outputs[idx][0]
                boxes = outputs[idx + fmc][0]
                boxes = boxes * stride
            else:
                scores = outputs[idx]
                boxes = outputs[idx + fmc]
                boxes = boxes * stride

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1)
                anchor_centers = anchor_centers.astype(np.float32)

                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1)
                    anchor_centers = anchor_centers.reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_indices = np.where(scores >= score_thresh)[0]
            bboxes = distance2box(anchor_centers, boxes)
            pos_scores = scores[pos_indices]
            pos_bboxes = bboxes[pos_indices]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
        return scores_list, bboxes_list

    def detect(self, image, input_size=None, score_threshold=0.5, max_num=0, metric='default'):
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size
        image_ratio = float(image.shape[0]) / image.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if image_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / image_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * image_ratio)
        det_scale = float(new_height) / image.shape[0]
        resized_img = cv2.resize(image, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list = self.forward(det_img, score_threshold)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if 0 < max_num < det.shape[0]:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = image.shape[0] // 2, image.shape[1] // 2
            offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                 (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            index = np.argsort(values)[::-1]  # some extra weight on the centering
            index = index[0:max_num]
            det = det[index, :]
        return det

    def nms(self, outputs):
        thresh = self.nms_thresh
        x1 = outputs[:, 0]
        y1 = outputs[:, 1]
        x2 = outputs[:, 2]
        y2 = outputs[:, 3]
        scores = outputs[:, 4]

        order = scores.argsort()[::-1]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            indices = np.where(ovr <= thresh)[0]
            order = order[indices + 1]

        return keep
