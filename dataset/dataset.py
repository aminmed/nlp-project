import torch
from torch.utils.data import Dataset

MAX_LEN = 40


class QuoraDataset(Dataset):

    def __init__(self, data, tokenizer, max_len = MAX_LEN ):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len
        self.targets = data['is_duplicate'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        question1 = str(self.data.iloc[index]['question1'])
        question2 = str(self.data.iloc[index]['question2'])
        label = self.data.iloc[index]['is_duplicate']
        
       ### [CLS] question1 [SEP] questions2 [SEP] ... [PAD]
        inputs = self.tokenizer.encode_plus(
            question1,
            question2,
            add_special_tokens=True,
            padding='max_length',
            max_length=2 * self.max_len + 3, # max length of 2 questions and 3 special tokens
            truncation=True   
        )
        
        # return targets 0, when using data set in testing and targets are none
        return {
            "ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(inputs["token_type_ids"], dtype=torch.long),
            "targets": torch.tensor(int(label), dtype=torch.long) if self.targets is not None else 0
        }


def get_data_loader(df, batch_size, shuffle, tokenizer):
    """
    Creates a dataset from a pandas DataFrame and returns a DataLoader object of the dataset.

    Args:
        df (pandas DataFrame): The DataFrame containing the question pairs.

        targets (numpy array): An array of binary target values indicating 
                                whether the question pairs are duplicates or not.

        batch_size (int): The size of each batch in the DataLoader.

        shuffle (bool): Whether to shuffle the data in the DataLoader.

        tokenizer (transformers tokenizer): The tokenizer to use for tokenizing the question pairs.

    Returns:
    
        data_loader (torch DataLoader): A DataLoader object containing the BertDataSet dataset.
    """

    # Create BertDataSet dataset from the DataFrame
    dataset = QuoraDataset(
        data = df,
        tokenizer=tokenizer
    )

    # Create DataLoader object from the dataset
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle=shuffle
    )

    return data_loader
