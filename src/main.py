import argparse
from chatbot import chatbot

def main():
    parser = argparse.ArgumentParser(description="Launch Chatbot with configurable parameters.")
    parser.add_argument("--sampling", type=str, default="classic",
                        choices=["diff", "classic"], help="Sampling method to use.")
    parser.add_argument("--mode", type=str, default="sentiment",
                        choices=["sentiment", "LLM"], help="sentiment or LLM scoring")
    parser.add_argument("--sentiment_label", type=int, default=1,
                        choices=[0, 1], help="1 - positive, 0 - negative")
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                        help="Name or path of the model to load.")
    parser.add_argument("--gen_length", type=int, default=64,
                        help="Generation length (number of tokens).")
    parser.add_argument("--block_length", type=int, default=32,
                        help="Block length for generation.")
    parser.add_argument("--steps", type=int, default=64,
                        help="Number of generation steps.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature.")
    parser.add_argument("--cfg_scale", type=float, default=0.0,
                        help="CFG scale parameter.")
    parser.add_argument("--score_prompt", type=str, default="Think Carefully", help="Prompt that will be used by score model")

    args = parser.parse_args()

    chatbot(sampling=args.sampling,
            mode=args.mode,
            sentiment_label=args.sentiment_label,
            model_name=args.model_name,
            gen_length=args.gen_length,
            block_length=args.block_length,
            steps=args.steps,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            score_prompt=args.score_prompt)

if __name__ == "__main__":
    main()
