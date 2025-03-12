import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from models import CheckpointedLLaDAModelLM
from generation import differentiable_generation, classic_generation
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

def chatbot(sampling="classic", mode="sentiment", sentiment_label=1, model_name="GSAI-ML/LLaDA-8B-Instruct",
            gen_length=64, block_length=32, steps=64, temperature=0.0, cfg_scale=0.0,
            sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english", score_prompt="Think Carefully"):
    """
    Chatbot loop with configurable generation parameters.
    
    This function initializes the language model and tokenizer, sets up the generation 
    method (differentiable or classic), and enters an interactive loop. The user input 
    is used to build a conversation prompt, and the model generates a response using the 
    specified generation method. The progress callback is used to provide live progress updates.
    
    Parameters:
      - sampling: "diff" for differentiable generation, any other value for classic generation.
      - model_name: Name or path of the pre-trained model.
      - gen_length: Number of tokens to generate for the response.
      - block_length: Number of tokens processed together as a block.
      - steps: Total generation steps (must align with block configuration).
      - temperature: Temperature value for stochastic sampling (via Gumbel noise).
      - cfg_scale: Classifier-free guidance scale.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )


    # If differentiable sampling is selected, wrap the model with checkpointing logic.
    if sampling == "diff":
        if mode == "sentiment":
            score_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name, torch_dtype=torch.bfloat16).to(device)
            score_model.gradient_checkpointing_enable()
            score_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
        else:
            model = CheckpointedLLaDAModelLM(model)
            score_model = None
            score_tokenizer = None

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
        
        # Generate a response using the selected sampling method.
        if sampling == "diff":
            final_x, _ = differentiable_generation(
                model=model,
                tokenizer=tokenizer,
                prompt_ids = prompt_ids,
                mode=mode,
                sentiment_label=sentiment_label,
                score_prompt=score_prompt,
                gen_length=gen_length,
                block_length=block_length,
                steps=steps,
                score_model=score_model,
                score_tokenizer=score_tokenizer,
                score_weight=1.0,
                lr=1e-2,
                temperature=temperature,
                remasking='low_confidence',
                progress_callback=live_progress_callback,
                cfg_scale=cfg_scale
            )
        else:
            final_x = classic_generation(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_ids,
                gen_length=gen_length,
                block_length=block_length,
                steps=steps,
                temperature=temperature,
                remasking='low_confidence',
                progress_callback=live_progress_callback,
                cfg_scale=cfg_scale
            )
        # Decode the generated tokens into text.
        generated_tokens = final_x[0, prompt_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        # Append the assistant's response to the conversation history.
        conversation.append({"role": "assistant", "content": output_text})
        print("")
        
        # If using differentiable generation, we exit after one response
        if sampling == "diff":
            break
