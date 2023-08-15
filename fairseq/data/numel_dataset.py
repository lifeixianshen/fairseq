# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from . import BaseWrapperDataset


class NumelDataset(BaseWrapperDataset):

    def __init__(self, dataset, reduce=False):
        super().__init__(dataset)
        self.reduce = reduce

    def __getitem__(self, index):
        item = self.dataset[index]
        return torch.numel(item) if torch.is_tensor(item) else np.size(item)

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return sum(samples) if self.reduce else torch.tensor(samples)
