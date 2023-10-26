import torch, logging
from transformers import LlamaTokenizer, LlamaForCausalLM

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

## v2 models
model_path = 'openlm-research/open_llama_3b_v2'
# model_path = 'openlm-research/open_llama_7b_v2'

## v1 models
# model_path = 'openlm-research/open_llama_3b'
# model_path = 'openlm-research/open_llama_7b'
# model_path = 'openlm-research/open_llama_13b'

logging.info(f'Loading model from {model_path}')

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map='auto',
).to(DEVICE)

logging.info('Model loaded')

def run(prompt, max_new_tokens=32):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    generation_output = model.generate(
        input_ids=input_ids, max_new_tokens=max_new_tokens
    )
    
    return tokenizer.decode(generation_output[0])