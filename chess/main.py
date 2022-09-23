import torch
import random
import os
from pascal import PascalVOC
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import (
    to_tensor,
    to_pil_image,
    convert_image_dtype,
)
from torchvision.ops import box_iou
from torchvision.utils import draw_bounding_boxes
from sklearn.preprocessing import LabelEncoder
import numpy as np

op = os.path


def mAP(ann, pred, device):
    boxes = pred["boxes"][pred["scores"] > 0.5]
    labels = pred["labels"][pred["scores"] > 0.5]
    ap = []
    cats = ann["labels"].unique()
    thresh = torch.arange(0.5, 1, 0.05)
    for cat in cats:
        y_true = ann["boxes"][ann["labels"] == cat]
        y_pred = boxes[labels == cat]
        iou = box_iou(y_pred, y_true.to(device))
        if len(iou):
            mask = iou.expand(len(thresh), *iou.shape) > thresh.expand(
                *iou.shape[::-1], len(thresh)
            ).T.to(device)
            tp = mask.sum(-1).sum(-1)
            fp = (mask.sum(-1) == 0).sum(-1)
            fn = (mask.sum(1) == 0).sum(-1)
            _ap = ((tp / (tp + fp + fn)).sum() / 10).cpu()
        else:
            _ap = 0

        ap.append(_ap)
    return np.mean(ap)


class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, root, df=None):  # , transforms=None):
        self._cache = {}
        self.root = root
        # self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        if df is None:
            self.imgs = list(
                sorted([f for f in os.listdir(root) if f.endswith(".jpg")])
            )
            self.masks = list(
                sorted([f for f in os.listdir(root) if f.endswith(".xml")])
            )
        labels = set()
        for mask in self.masks:
            ann = PascalVOC.from_xml(op.join(self.root, mask))
            labels.update([k.name for k in ann.objects])
        self.lenc = LabelEncoder().fit(list(labels))

    def get_mask(self, idx):
        ann = PascalVOC.from_xml(op.join(self.root, self.masks[idx]))
        return [
            (k.name, k.bndbox.xmin, k.bndbox.ymin, k.bndbox.xmax, k.bndbox.ymax)
            for k in ann.objects
        ]

    def draw_bbox(
        self,
        idx,
        model=None,
        ax=None,
        device=False,
        pred_thresh=0.5,
        title=True,
        **kwargs,
    ):
        img, ann = self[idx]
        if device:
            img = img.to(device)
        labels_true = self.lenc.inverse_transform(ann["labels"])
        drawn = draw_bounding_boxes(
            convert_image_dtype(img, torch.uint8),
            boxes=ann["boxes"],
            labels=labels_true,
            colors="green",
            width=2,
            font="Ubuntu-R.ttf",
            font_size=20,
        )
        if model is not None:
            # Add predictions to the drawn model
            pred = model([img])[0]
            pred["labels"] = pred["labels"][pred["scores"] > pred_thresh]
            pred["boxes"] = pred["boxes"][pred["scores"] > pred_thresh]
            labels_pred = self.lenc.inverse_transform(pred["labels"])
            drawn = draw_bounding_boxes(
                drawn,
                boxes=pred["boxes"],
                labels=labels_pred,
                colors="red",
                width=2,
                font="Ubuntu-R.ttf",
                font_size=20,
            )
        plt.imshow(to_pil_image(drawn))
        if title:
            plt.title(f"{idx}")
        plt.show()

    def plot_sample(self, nrows=2, ncols=2, annot=True, idx=None, **kwargs):
        if idx is None:
            n_samples = nrows * ncols
            idx = list(range(len(self.imgs)))
            random.shuffle(idx)
            idx = idx[:n_samples]

        fig, ax = plt.subplots(nrows, ncols, **kwargs)
        try:
            ax = ax.ravel()
        except AttributeError:
            ax = [ax]
        for _ax, im_id in zip(ax, idx):
            img, ann = self[im_id]
            img = to_pil_image(img)
            _ax.imshow(img)
            labels = self.lenc.inverse_transform([k.item() for k in ann["labels"]])
            for (box, label) in zip(ann["boxes"], labels):
                xmin, ymin, xmax, ymax = box
                _ax.vlines([xmin, xmax], ymax, ymin, colors="g")
                _ax.hlines([ymax, ymin], xmin, xmax, color="g")
                if annot:
                    _ax.text(xmin, ymin, label, {"fontsize": "xx-small"})
            _ax.set_axis_off()

        plt.tight_layout()
        plt.show()

    def __getitem__(self, idx):
        # load images ad masks
        if idx in self._cache:
            return self._cache[idx]
        img_path = op.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        masks = self.get_mask(idx)

        boxes = [k[1:] for k in masks]
        labels = [k[0] for k in masks]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {"id": image_id}
        target["boxes"] = boxes
        target["labels"] = torch.tensor(self.lenc.transform(labels))
        target["area"] = area

        img = to_tensor(img)
        self._cache[idx] = img, target

        return img, target

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    root = op.join(op.dirname(__file__), "data/train")
    dataset = ChessDataset(root)
    model = torch.load("data/models/strat.pt").eval().cpu()
    for idx in [36, 172, 38, 160, 121, 74, 15, 113, 138, 77, 194, 0, 1, 178, 97]:
        dataset.draw_bbox(idx, model)
