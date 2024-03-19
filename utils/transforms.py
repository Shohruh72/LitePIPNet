import cv2
import random
import numpy as np
from PIL import Image, ImageFilter


def resample():
    return random.choice((Image.NEAREST, Image.BILINEAR, Image.BICUBIC))


class RandomTranslate:
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            h, w = image.size
            a = 1
            b = 0
            c = int((random.random() - 0.5) * 60)
            d = 0
            e = 1
            f = int((random.random() - 0.5) * 60)
            image = image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f), resample=resample())
            label = label.copy()
            label = label.reshape(-1, 2)
            label[:, 0] -= 1. * c / w
            label[:, 1] -= 1. * f / h
            label = label.flatten()
            label[label < 0] = 0
            label[label > 1] = 1
            return image, label
        else:
            return image, label


class RandomRotate:
    def __init__(self, angle=45, p=0.5):
        self.angle = angle
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            num_lms = int(len(label) / 2)

            center_x = 0.5
            center_y = 0.5

            label = np.array(label) - np.array([center_x, center_y] * num_lms)
            label = label.reshape(num_lms, 2)
            theta = random.uniform(-np.radians(self.angle), +np.radians(self.angle))
            angle = np.degrees(theta)
            image = image.rotate(angle, resample=resample())

            cos = np.cos(theta)
            sin = np.sin(theta)
            label = np.matmul(label, np.array(((cos, -sin), (sin, cos))))
            label = label.reshape(num_lms * 2) + np.array([center_x, center_y] * num_lms)
            return image, label
        else:
            return image, label


class RandomFlip:
    def __init__(self, params, p=0.5):
        self.points_flip = (np.array(params['points_flip']) - 1).tolist()
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = np.array(label).reshape(-1, 2)
            label = label[self.points_flip, :]
            label[:, 0] = 1 - label[:, 0]
            label = label.flatten()
            return image, label
        else:
            return image, label


class RandomCutOut:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = np.array(image).astype(np.uint8)
            image = image[:, :, ::-1]
            h, w, _ = image.shape
            cut_h = int(h * 0.4 * random.random())
            cut_w = int(w * 0.4 * random.random())
            x = int((w - cut_w - 10) * random.random())
            y = int((h - cut_h - 10) * random.random())
            image[y:y + cut_h, x:x + cut_w, 0] = int(random.random() * 255)
            image[y:y + cut_h, x:x + cut_w, 1] = int(random.random() * 255)
            image[y:y + cut_h, x:x + cut_w, 2] = int(random.random() * 255)
            image = Image.fromarray(image[:, :, ::-1].astype('uint8'), 'RGB')
            return image, label
        else:
            return image, label


class RandomGaussianBlur:
    def __init__(self, p=0.75):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            radius = random.random() * 5
            gaussian_blur = ImageFilter.GaussianBlur(radius)
            image = image.filter(gaussian_blur)
        return image, label


class RandomHSV:
    def __init__(self, h=0.015, s=0.700, v=0.400, p=0.500):
        self.h = h
        self.s = s
        self.v = v
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = np.array(image)
            r = np.random.uniform(-1, 1, 3)
            r = r * [self.h, self.s, self.v] + 1
            hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype('uint8')
            lut_sat = np.clip(x * r[1], 0, 255).astype('uint8')
            lut_val = np.clip(x * r[2], 0, 255).astype('uint8')

            image_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB, dst=image)
            image = Image.fromarray(image)
        return image, label


class RandomRGB2IR:
    """
    RGB to IR conversion
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() > self.p:
            return image, label
        image = np.array(image)
        image = image.astype('int32')
        delta = np.random.randint(10, 90)

        ir = image[:, :, 2]
        ir = np.clip(ir + delta, 0, 255)
        return Image.fromarray(np.stack((ir, ir, ir), axis=2).astype('uint8')), label
