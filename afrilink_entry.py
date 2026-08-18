from pathlib import Path
import runpy
import shutil
import sys
import tarfile
import torch

input_dir = Path("/workspace/job/input")
combined_archive = Path("/tmp/afrilink_training.tar.gz")

# find and reassemble the uploaded pieces # 413 eror rupload size
parts = sorted(input_dir.rglob("training.tar.gz.part-*"))

if not parts:
    raise RuntimeError("No dataset archive pieces were uploaded.")

print(f"Reassembling dataset from {len(parts)} pieces...", flush=True)

with combined_archive.open("wb") as output:
    for part in parts:
        with part.open("rb") as source:
            shutil.copyfileobj(source, output)

print("Extracting complete dataset...", flush=True)

with tarfile.open(combined_archive, "r:gz") as archive:
    archive.extractall(input_dir)

train_script = input_dir / "src" / "train.py"
train_data = input_dir / "datasets" / "encoded" / "train_vocab_4000.npy"
validation_data = input_dir / "datasets" / "encoded" / "valid_vocab_4000.npy"

for required_file in (train_script, train_data, validation_data):
    if not required_file.exists():
        raise RuntimeError(f"Required file was not extracted: {required_file}")

print(f"Training script: {train_script}", flush=True)
print(f"Training data: {train_data}", flush=True)
print(f"Validation data: {validation_data}", flush=True)

# confirm the assigned GPU make sure it only runs cuda and nothing else
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Refusing to train.")

gpu_name = torch.cuda.get_device_name(0)
gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

print("=" * 60, flush=True)
print(f"GPU: {gpu_name}", flush=True)
print(f"GPU memory: {gpu_memory:.1f} GB", flush=True)
print(f"CUDA version: {torch.version.cuda}", flush=True)
print(f"BF16 supported: {torch.cuda.is_bf16_supported()}", flush=True)
print("=" * 60, flush=True)

if "L4" not in gpu_name.upper():
    raise RuntimeError(f"Expected an NVIDIA L4, received: {gpu_name}")

if not torch.cuda.is_bf16_supported():
    raise RuntimeError("The assigned GPU does not support BF16.")

# afriLink adds --data for the entry script train py donest
while "--data" in sys.argv:
    argument_index = sys.argv.index("--data")
    del sys.argv[argument_index : argument_index + 2]

print(f"Training arguments: {sys.argv[1:]}", flush=True)
sys.path.insert(0, str(input_dir))
runpy.run_path(str(train_script),run_name="__main__",)