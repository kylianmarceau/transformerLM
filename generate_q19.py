from pathlib import Path
import torch
from src.evaluate import model_from_checkpoint
from src.generate import generate
from src.tokenizer import BPETokenizer, END_OF_TEXT

PROJECT_ROOT = Path(__file__).resolve().parent
PROMPT = "Once upon a time, "
SEED = 3043

CHECKPOINT = (PROJECT_ROOT/ "afrilink_results/final/final_vocab4000_20000/checkpoints"/ "final_vocab4000_20000_step_020000.pt")

SETTINGS = [
    {"name": "Top-p sampling","temperature": 0.8,"top_k": None,"top_p": 0.9,},
    {"name": "Top-k sampling","temperature": 0.8,"top_k": 40,"top_p": None,},
    {"name": "Greedy decoding","temperature": 0.0,"top_k": None,"top_p": None,},
]


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Loading final model on {device}...")
    model, _ = model_from_checkpoint(CHECKPOINT, device)

    tokenizer = BPETokenizer.from_files(PROJECT_ROOT / "tokenizers/vocab_4000/vocab.tsv",PROJECT_ROOT / "tokenizers/vocab_4000/merges.tsv",[END_OF_TEXT],)

    samples = []

    for number, setting in enumerate(SETTINGS, start=1):
        print(f"Generating sample {number}: {setting['name']}...")

        text = generate(model=model,tokenizer=tokenizer,prompt=PROMPT,max_new_tokens=256,temperature=setting["temperature"],top_k=setting["top_k"],top_p=setting["top_p"],seed=SEED,use_cache=True,)

        samples.append(
            f"Q19 SAMPLE {number}: {setting['name']}\n"
            f"Prompt: {PROMPT!r}\n"
            f"Temperature: {setting['temperature']}\n"
            f"Top-k: {setting['top_k']}\n"
            f"Top-p: {setting['top_p']}\n"
            f"Seed: {SEED}\n"
            f"Maximum new tokens requested: 256\n\n"
            f"{text}\n"
        )

    note = ("Note: the model context length is 256 tokens and the prompt uses " "6 tokens, so the implementation can generate at most 250 new tokens.\n\n")

    output = PROJECT_ROOT / "samples.txt"
    output.write_text(note + ("\n" + "-" * 80 + "\n\n").join(samples),encoding="utf-8",)

    print(f"Done. Samples saved to {output}")


if __name__ == "__main__":
    main()