from torch.utils.data import Dataset

from ..utils import common_functions as c_f


class DomainDataset(Dataset):
    def __init__(self, dataset: Dataset, domain: str, domain_idx: int):
        self.dataset = dataset
        self.domain = domain
        self.domain_idx = domain_idx

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        return c_f.nice_repr(
            self, c_f.extra_repr(self, ["domain"]), {"dataset": self.dataset}
        )
    
    def __getitem__(self, index):
        item = self.dataset[index]
        if isinstance(item, tuple):
            return item + (self.domain,)
        else:
            return self.dataset[index], self.domain
