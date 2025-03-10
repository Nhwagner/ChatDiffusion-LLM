import torch
from transformers import AutoTokenizer, AutoModel
from models import CheckpointedLLaDAModelLM
from generation import differentiable_generation, classical_generation
from utils import live_progress_callback
import time

def build_chat_prompt(conversation, add_generation_prompt=True):
    """
    Build a text prompt from the conversation history.
    """
    prompt = ""
    for msg in conversation:
        role = msg["role"].capitalize()
        content = msg["content"]
        prompt += f"{role}: {content}\n"
    if add_generation_prompt:
        prompt += "Assistant: "
    return prompt

def chatbot():
    """
    Chatbot loop with a clean display.
    """
    sampling = "diff"  # Choose between "diff" or "classical"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = AutoModel.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct',
        trust_remote_code=True
    )
    
    if sampling == "diff":
        model = CheckpointedLLaDAModelLM(model)

    conversation = []
    print("Chatbot initialized. Type 'exit' or 'quit' to stop.")

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        conversation.append({"role": "user", "content": user_input})
        prompt_text = build_chat_prompt(conversation, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

        # Reset callback state for new generation
        if hasattr(live_progress_callback, "first_call"):
            delattr(live_progress_callback, "first_call")
        live_progress_callback.first_call = True
        live_progress_callback.last_len = 0

        if sampling == "diff":
            final_x, _ = differentiable_generation(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                gen_length=32,
                block_length=8,
                steps=16,
                score_model=model,
                score_weight=1.0,
                lr=1e-2,
                temperature=0.,
                remasking='low_confidence',
                progress_callback=live_progress_callback,
                cfg_scale=0.
            )
        else:
            final_x = classical_generation(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_ids,
                gen_length=64,
                block_length=32,
                steps=64,
                temperature=0.,
                remasking='low_confidence',
                progress_callback=live_progress_callback,
                cfg_scale=0.
            )
        
        generated_tokens = final_x[0, prompt_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": output_text})
        print("")
        if sampling == "diff":
            break
