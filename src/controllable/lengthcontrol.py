import torch
import argparse
from transformers import BertModel, BertTokenizer

def load_model_and_tokenizer(model_path):
    # Load the BERT model architecture
    model = BertModel.from_pretrained('bert-base-uncased')  # Adjust model type if necessary
    # Load the custom checkpoint
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Adjust model type if necessary
    return model, tokenizer

def generate_text_with_bert(model, tokenizer, prompt, max_length):
    # Encoding the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt', max_length=max_length, truncation=True)
    attention_mask = torch.ones(input_ids.shape)

    # Getting the output from BERT
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # Getting the last hidden states
    last_hidden_states = outputs.last_hidden_state

    # Decoding the output (this is a naive approach and may not produce coherent text)
    decoded_output = tokenizer.decode(last_hidden_states[0], skip_special_tokens=True)
    return decoded_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate text using a BERT model.')
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Model name or path')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt to start text generation')
    parser.add_argument('--max_length', type=int, default=50, help='Maximum length of the generated text')

    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)

    # Generate text
    generated_text = generate_text_with_bert(model, tokenizer, args.prompt, args.max_length)
    print(generated_text)
