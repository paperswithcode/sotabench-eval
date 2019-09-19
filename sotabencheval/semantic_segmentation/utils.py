import numpy as np


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        """
        print(a.shape)
        print(n.shape)
        k = (a >= 0) & (a < n)
        inds = n * a[k].to(torch.int64) + b[k]
        self.mat += np.bincount(inds, minlength=n ** 2).reshape(n, n)
        """
        n = self.num_classes

        if self.mat is None:
            self.mat = np.zeros((n, n), dtype=np.int64)

        k = (a >= 0) & (a < n)
        inds = n * a[k].astype(np.int64) + b[k]
        self.mat += np.bincount(inds, minlength=n ** 2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat
        acc_global = np.diag(h).sum() / h.sum()
        acc = np.diag(h) / h.sum(1)
        iu = np.diag(h) / (h.sum(1) + h.sum(0) - np.diag(h))
        return acc_global, acc, iu

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            "global correct: {:.1f}\n"
            "average row correct: {}\n"
            "IoU: {}\n"
            "mean IoU: {:.1f}"
        ).format(
            acc_global.item() * 100,
            ["{:.1f}".format(i) for i in (acc * 100).tolist()],
            ["{:.1f}".format(i) for i in (iu * 100).tolist()],
            iu.mean().item() * 100,
        )

