# ChatDiffusion-LLM
Diffusion-based differentiable generation for LLM synthesis.

This repository provides an interactive simple chat interface built on top of the LLaDA model to support live non-sequential sampling responses in the terminal. Supports classic LLaDA sampling and a novel method for differentiable generation incorporating a score model. The implementation uses PyTorch and Hugging Face’s transformers library with configurable parameters to experiment with various generation strategies.

## Usage

Launch the script with the desired configuration. For example:

```bash
python src/main.py --sampling classic --model_name GSAI-ML/LLaDA-8B-Instruct --gen_length 64 --block_length 32 --steps 64 --temperature 0.0 --cfg_scale 0.0
