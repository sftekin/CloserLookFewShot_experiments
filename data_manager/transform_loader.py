import torch
from torchvision.transforms import transforms
from PIL import ImageEnhance


class TransformLoader:
    def __init__(self, image_size, augmentation=False):
        self.image_size = image_size
        self.norm_param = {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
        self.jitter_param = {
            "Brightness": 0.4,
            "Contrast": 0.4,
            "Color": 0.4
        }

        if not augmentation:
            self.transform_list = ['Resize', 'CenterCrop', 'ToTensor', 'Normalize']
        else:
            self.transform_list = ['RandomSizedCrop', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']

    def parse_transform(self, transform_type):
        if transform_type == 'ImageJitter':
            method = ImageJitter(self.jitter_param)
            return method
        method = getattr(transforms, transform_type)
        if transform_type == 'RandomSizedCrop':
            return method(self.image_size)
        elif transform_type == 'CenterCrop':
            return method(self.image_size)
        elif transform_type == 'Resize':
            return method([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        elif transform_type == 'Normalize':
            return method(**self.norm_param)
        else:
            return method()

    def get_transform(self):
        transform_funcs = [self.parse_transform(x) for x in self.transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform


class ImageJitter(object):
    def __init__(self, transform_dict):
        transform_type = dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast,
                              Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)

        self.transforms = [(transform_type[k], transform_dict[k]) for k in transform_dict]

    def __call__(self, img):
        out = img
        rand_tensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (rand_tensor[i] * 2.0 - 1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out
