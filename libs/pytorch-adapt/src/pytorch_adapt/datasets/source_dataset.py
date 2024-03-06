import torch
from typing import Any, Dict, List
from torch.utils.data import Dataset
from .domain_dataset import DomainDataset

# single source dataset
class SourceDataset(DomainDataset):
    """
    Wrap your source dataset with this. Your source dataset's
    ```__getitem__``` function should return a tuple of ```(data, label)```.
    """

    def __init__(self, dataset: Dataset, domain: int = 0, domain_idx: int = 0):
        """
        Arguments:
            dataset: The dataset to wrap
            domain: An integer representing the domain.
        """
        super().__init__(dataset, domain, domain_idx)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns:
            A dictionary with keys

                - "src_imgs" (the data)

                - "src_domain" (the integer representing the domain)

                - "src_labels" (the class label)

                - "src_sample_idx" (idx)
        """

        img, src_labels = self.dataset[idx]
        return {
            "src_imgs": img,
            "src_domain": self.domain,
            "src_domain_idx": self.domain_idx,
            "src_labels": src_labels,
            "src_sample_idx": idx,
        }


# class MultiDataset(Dataset):
#     """
#     Wrap your source dataset with this. Your source dataset's
#     ```__getitem__``` function should return a tuple of ```(data, label)```.
#     """

#     def __init__(self, datasets: List[Dataset], domains: List[int]):
#         """
#         Arguments:
#             datasets: The datasets to wrap
#         """
#         super().__init__(datasets)
#         self.datasets = datasets
#         self.domains = domains
#         # shuffle datasets
#         perm = torch.randperm(len(self.datasets))
#         self.datasets = [self.datasets[idx] for idx in perm]
#         self.domains = [self.domains[idx] for idx in perm]

#     def __getitem__(self, idx: int) -> Dict[str, Any]:
#         """
#         Returns:
#             A dictionary with keys

#                 - "src_imgs" (the data)

#                 - "src_domain" (the integer representing the domain)

#                 - "src_labels" (the class label)

#                 - "src_sample_idx" (idx)
#         """

#         img, src_labels = self.dataset[idx]
#         domain = self.domains[idx]
#         return {
#             "src_imgs": img,
#             "src_domain": domain,
#             "src_labels": src_labels,
#             "src_sample_idx": idx,
#         }
