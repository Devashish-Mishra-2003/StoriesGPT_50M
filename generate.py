import torch
from tokenizers import Tokenizer
from model import ModernGPT
from config import GPTConfig
from huggingface_hub import hf_hub_download

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer (local file)
tokenizer = Tokenizer.from_file("tokenizer.json")

# Build model
config = GPTConfig()
model = ModernGPT(config).to(device)

# Download weights from HuggingFace
ckpt_path = hf_hub_download(
    repo_id="devashishmishr/StoriesGPT-50M",
    filename="model_final.pt"
)

# Load checkpoint properly
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def clean_text(text):
    text = text.replace("Ġ", " ")
    text = text.replace("Ċ", "\n")
    return text.strip()


def generate(prompt, max_new_tokens=150):
    ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        input_ids = torch.cat([input_ids, next_token], dim=1)

    return clean_text(tokenizer.decode(input_ids[0].tolist()))


if __name__ == "__main__":
    prompt = input("Enter prompt: ")
    print(generate(prompt))