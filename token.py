from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Initialize a tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Define special tokens
special_tokens = ["[UNK]", "[PAD]", "[START]", "[END]"]

# Configure the pre-tokenizer to split on whitespace
tokenizer.pre_tokenizer = Whitespace()

# Initialize the trainer
trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=25000)

# Prepare your files
files = ["path/to/file1.txt", "path/to/file2.txt", ...]  # Add your data files here

# Train the tokenizer
tokenizer.train(files, trainer)

# Save the tokenizer
tokenizer.save("path/to/save/tokenizer.json")

# Example usage
text = "Hello, how are you today?"
encoded = tokenizer.encode(text)
print(f"Encoded: {encoded.ids}")
print(f"Decoded: {tokenizer.decode(encoded.ids)}")
