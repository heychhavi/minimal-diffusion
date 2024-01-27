import torch
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')  # Load GPT-2 architecture
model.load_state_dict(torch.load('ckpts/greetings/ema_0.9999_000500.pt'))  # Load your custom weights

def generate_text_of_specific_length(model_name, target_length, prompt):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=target_length, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate text of a specific length using a GPT-2 model.')
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Model name or path')
    parser.add_argument('--target_length', type=int, required=True, help='Target length of generated text')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt to start text generation')
    
    args = parser.parse_args()

    generated_text = generate_text_of_specific_length(args.model_name_or_path, args.target_length, args.prompt)
    print(generated_text)
