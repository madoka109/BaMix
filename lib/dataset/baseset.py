from torch.utils.data import Dataset
import json, os, time
import cv2


class BaseSet(Dataset):
    def __init__(self, mode="train", transform=None, args=None):
        self.mode = mode
        self.transform = transform
        self.args = args
        self.color_space = self.args.COLOR_SPACE

        print("Use {} Mode to train network".format(self.color_space))

        if self.mode == "train":
            print("Loading train data ...", end=" ")
            self.json_path = self.args.TRAIN_JSON
        elif "valid" in self.mode:
            print("Loading valid data ...", end=" ")
            self.json_path = self.args.VALID_JSON
        else:
            raise NotImplementedError

        # read json file
        with open(self.json_path, "r") as f:
            self.all_info = json.load(f)
        self.num_classes = self.all_info["num_classes"]
        self.data = self.all_info['annotations']

        print("Contain {} images of {} classes".format(len(self.data), self.num_classes))

        self.class_dict = self._get_class_dict()

    def __len__(self):
        return len(self.all_info['annotations'])

    def __getitem__(self, index):
        print('start get item...')
        now_info = self.data[index]
        img = self._get_image(now_info)
        print('complete get img...')
        meta = dict()
        image = self.transform(img)
        image_label = (
            now_info["category_id"] if "test" not in self.mode else 0
        )  # 0-index
        if self.mode not in ["train", "valid"]:
           meta["image_id"] = now_info["image_id"]
           meta["fpath"] = now_info["fpath"]

        return image, image_label, meta

    def get_num_classes(self):
        return self.num_classes

    def get_annotations(self):
        return self.all_info['annotations']

    def imread_with_retry(self, fpath):
        retry_time = 10
        for k in range(retry_time):
            try:
                img = cv2.imread(fpath)
                if img is None:
                    print(f"img{fpath} is None, try to re-read img")
                    continue
                return img
            except Exception as e:
                if k == retry_time - 1:
                    assert False, "pillow open {} failed".format(fpath)
                time.sleep(0.1)

    def _get_image(self, now_info):
        fpath = os.path.join(now_info["fpath"])
        img = self.imread_with_retry(fpath)
        if self.color_space == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.data):
            cat_id = (
                anno["category_id"] if "category_id" in anno else anno["image_label"]
            )
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_cls_num_list(self):
        for i in range(len(self.class_dict)):
            self.cls_num_list.append(len(self.class_dict[i]))
        return self.cls_num_list


