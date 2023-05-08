import torch
from torch.utils.data import Dataset

class QuoraDataset(Dataset):

    def __init__(self, data, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        question1 = str(self.data.iloc[index]['question1'])
        question2 = str(self.data.iloc[index]['question2'])
        label = self.data.iloc[index]['is_duplicate']
        
        encoding = self.tokenizer.encode_plus(
            question1,
            question2,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            return_attention_mask=True,
            truncation=True
        )

        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(encoding['token_type_ids'], dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float)
        }


def get_data_loader(df, targets, batch_size, shuffle, tokenizer):
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
        first_questions=df["question1"].values,
        second_questions=df["question2"].values,
        targets=targets,
        tokenizer=tokenizer
    )

    # Create DataLoader object from the dataset
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle=shuffle
    )

    return data_loader
