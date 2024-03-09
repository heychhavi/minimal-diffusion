import torch
from torch.utils.data import Dataset, DataLoader

def load_sentences_and_scores(file_path, tokenizer, max_seq_len):
    sentences = []
    scores = []
    tokenized_texts = []
    attention_masks = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Trim the line to remove leading/trailing whitespace
            line = line.strip()
            # Attempt to split the line into a sentence and score
            parts = line.split('|')
            if len(parts) == 2:
                sentence, score_str = parts
                try:
                    # Attempt to convert the score to a float
                    score = float(score_str)
                except ValueError:
                    # If conversion fails, log the error and skip this line
                    print(f"Could not convert score to float for line: {line}")
                    continue
                
                sentences.append(sentence.strip())
                scores.append(score)

                # Tokenize and encode the sentence
                encoded_dict = tokenizer.encode_plus(
                    sentence.strip(),
                    add_special_tokens=True,
                    max_length=max_seq_len,
                    padding='max_length',
                    return_attention_mask=True,
                    return_tensors='pt',
                    truncation=True
                )
                
                tokenized_texts.append(encoded_dict['input_ids'])
                attention_masks.append(encoded_dict['attention_mask'])
            else:
                # If the line does not contain exactly one delimiter, log and skip it
                print(f"Skipping line due to unexpected format: {line}")

    if not tokenized_texts:
        raise RuntimeError("No valid data was loaded. Please check the input file and format.")

    tokenized_texts = torch.cat(tokenized_texts, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    scores = torch.tensor(scores)

    return sentences, scores, tokenized_texts, attention_masks



class CustomDataset(Dataset):
    """
    A custom PyTorch Dataset that takes a list of tokenized texts, attention masks, and concreteness scores.
    """
    def __init__(self, tokenized_texts, attention_masks, scores):
        self.tokenized_texts = tokenized_texts
        self.attention_masks = attention_masks
        self.scores = scores

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        return {
            "input_ids": self.tokenized_texts[idx],
            "attention_mask": self.attention_masks[idx],
            "concreteness_scores": self.scores[idx]
        }

def create_data_loader(tokenized_texts, attention_masks, scores, batch_size):
    """
    Creates a DataLoader from tokenized texts, attention masks, and concreteness scores.
    
    Parameters:
    - tokenized_texts: Tokenized and encoded sentences.
    - attention_masks: Attention masks for the tokenized_texts.
    - scores: Concreteness scores for the sentences.
    - batch_size: Batch size for the DataLoader.
    
    Returns:
    - A DataLoader object ready for iteration.
    """
    dataset = CustomDataset(tokenized_texts, attention_masks, scores)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader
