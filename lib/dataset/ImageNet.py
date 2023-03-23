from lib.dataset.baseset import BaseSet
import random


class ImageNet(BaseSet):
    def __init__(self, mode='train', transform=None, args=None):
        super(ImageNet, self).__init__(mode, transform, args=args)
        random.seed(0)
        self.cls_num_list = []

    def __getitem__(self, index):

        now_info = self.data[index]
        img = self._get_image(now_info)
        image = self.transform(img)
        image_label = now_info['category_id']  # 0-index
        return image, image_label













