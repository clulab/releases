from torch.utils.data import Dataset, DataLoader
import torch

class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, customized_tokenizer):
        '''
        :param customized_tokenizer: should be t5tokenizer with added special tokens
        '''
        self.customized_tokenizer = customized_tokenizer

    def pad_collate(self, batch):
        input_strings = [item["input"] for item in batch]
        output_strings = [item["output"] for item in batch]
        return {"input": self.customized_tokenizer(input_strings, padding=True, truncation=True, return_tensors="pt"),
                "output": self.customized_tokenizer(output_strings, padding=True, truncation=True, return_tensors="pt"),
                "input_strings": input_strings,
                "output_strings": output_strings}

    def __call__(self, batch):
        return self.pad_collate(batch)

class RuleTakerDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, instances_list):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.all_instances = instances_list

    def __len__(self):
        return len(self.all_instances)

    def __getitem__(self, idx):
        return self.all_instances[idx]


class PadCollateT5Standard:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, customized_tokenizer):
        '''
        :param customized_tokenizer: should be t5tokenizer with added special tokens
        '''
        self.customized_tokenizer = customized_tokenizer

    def pad_collate(self, batch):
        input_strings = ["context: "+item["context"]+" question: "+item["question"]+" </s>" for item in batch]
        output_strings = [item["answer"]+" </s>" for item in batch]

        return {"input": self.customized_tokenizer(input_strings, padding=True, truncation=True, return_tensors="pt"),
                "output": self.customized_tokenizer(output_strings, padding=True, truncation=True, return_tensors="pt")}

    def __call__(self, batch):
        return self.pad_collate(batch)

class RuleTakerDatasetT5Standard(Dataset):
    def __init__(self, instances_list):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.all_instances = instances_list

    def __len__(self):
        return len(self.all_instances)

    def __getitem__(self, idx):
        return self.all_instances[idx]