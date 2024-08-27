import torch
from transformers import AutoModel, AutoTokenizer

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model_path = "/home/sycao/Documents/FlexiBatch/models/gemma"
mdoel2_path = "/home/sycao/Documents/FlexiBatch/models/llama"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# input_path = "./data/ShareGPT_V3_unfiltered_cleaned_split.json"
input_text = "I am a scientist."
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs["input_ids"].to("cuda")

def manual_layer(input_ids, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    
    # model forward to 
    outputs = model(input_ids, output_hidden_states=True)
    layer_outputs = outputs.last_hidden_state
    for i, layer in enumerate(model.layers):
        layer_states = layer(layer_outputs[0][0])
        
        if i > 0:
            del model.layers[i - 1]
            torch.cuda.empty_cache()
            
            
        print(f"Output after layer{i + 1}:{layer_states.shape}")
        
    
    return layer_states

final_output = manual_layer(input_ids, model)

print(f"Final output is:{final_output.shape}")
    