import torch
import pandas as pd
import os
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


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annpath, intix=False):
        self._cache = {}
        self.root = root
        self.masks = pd.read_json(annpath)
        self.imgs = np.sort(self.masks["id"].unique())
        self.lenc = LabelEncoder()
        self.masks["label"] = self.lenc.fit_transform(self.masks["label"])
        self.intix = intix

    def get_mask(self, idx):
        df = self.masks[self.masks["id"] == idx][["label", "bbox"]]

        def _bbox_format(x, y, width, height):
            return x, y, x + width, y + height

        return df.apply(
            lambda x: (x["label"], *_bbox_format(*x["bbox"])), axis=1
        ).tolist()

    def draw_bbox(
        self,
        idx,
        model=None,
        ax=None,
        device=False,
        pred_thresh=0.5,
        show_iou=True,
        **kwargs,
    ):
        img, ann = self[idx]
        if not device:
            device = torch.device('cpu')
        img = img.to(device)
        labels_true = self.lenc.inverse_transform(ann["labels"])
        drawn = draw_bounding_boxes(
            convert_image_dtype(img, torch.uint8),
            boxes=ann["boxes"],
            labels=labels_true,
            colors="green",
            width=1,
            font="Ubuntu-R.ttf",
            font_size=10,
        )
        if model is not None:
            # Add predictions to the drawn model
            pred = model([img])[0]
            cutoff = pred['scores'] > pred_thresh
            for key in ['labels', 'boxes', 'scores']:
                pred[key] = pred[key][cutoff]
            labels_pred = self.lenc.inverse_transform(pred["labels"])
            drawn = draw_bounding_boxes(
                drawn,
                boxes=pred["boxes"],
                labels=labels_pred,
                colors="red",
                width=1,
                font="Ubuntu-R.ttf",
                font_size=10,
            )
        plt.imshow(to_pil_image(drawn))
        title = f"{idx}"
        if show_iou:
            title += f" ({round(mAP(ann, pred, device), 2)})"
        plt.title(title)
        plt.show()

    def plot_sample(self, nrows=2, ncols=2, annot=True, idx=None, **kwargs):
        if idx is None:
            n_samples = nrows * ncols
            newix = np.arange(len(self.imgs))
            np.random.shuffle(newix)
            idx = self.imgs[newix][:n_samples]

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
        if self.intix:
            idx = self.imgs[idx]
        if idx in self._cache:
            return self._cache[idx]
        img_path = op.join(self.root, f"{idx}".rjust(12, "0") + ".jpg")
        img = Image.open(img_path).convert("RGB")
        masks = self.get_mask(idx)

        boxes = [k[1:] for k in masks]
        labels = [k[0] for k in masks]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {"id": image_id}
        target["boxes"] = boxes
        target["labels"] = torch.tensor(labels)
        target["area"] = area

        img = to_tensor(img)
        self._cache[idx] = img, target

        return img, target

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    ds = COCODataset("data/train", "data/annotations.json")
    # bad_idx = [
    #     411832, 44781, 518685, 455135, 323853, 246725, 30932, 114504, 107167, 37863,
    #     246382, 1307, 456936, 558137, 416733, 1355, 221245, 281970, 311337, 81177
    # ]
    bad_idx = [107167, 518685, 44781, 246725, 30932, 114504, 323853, 411832,
               37863, 455135]
    model = torch.load("data/models/strat.pt").eval().cpu()
    # ds.draw_bbox(221245, model)
    for idx in bad_idx:
        ds.draw_bbox(idx, model)
