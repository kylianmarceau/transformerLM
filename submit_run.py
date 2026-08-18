from __future__ import annotations
import argparse
import os
from pathlib import Path
#sdk
from afrilink import AfriLinkClient

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--group", default="experiments")
parser.add_argument("--lr", required=True)
parser.add_argument("--steps", type=int, required=True)
parser.add_argument("--vocab-size", type=int, default=4000)
parser.add_argument("--warmup", type=int, default=200)
parser.add_argument("--seed", type=int, default=3043)
parser.add_argument("--rmsnorm", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--rope", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--ffn-type", choices=("swiglu", "relu"), default="swiglu")
parser.add_argument("--evaluation-interval", type=int, default=100)
parser.add_argument("--evaluation-batches", type=int, default=20)
parser.add_argument("--checkpoint-interval", type=int, default=500)
parser.add_argument("--time-limit", default="01:00:00")
parser.add_argument("--data", default="afrilink_training_data.tar.gz")

args = parser.parse_args()

train_path = (f"/workspace/job/input/datasets/encoded/" f"train_vocab_{args.vocab_size}.npy")
validation_path = (f"/workspace/job/input/datasets/encoded/" f"valid_vocab_{args.vocab_size}.npy")

script_args = ["--train-path",train_path,"--validation-path",validation_path,"--output-dir","/workspace/job/output","--run-name",args.name,"--learning-rate",args.lr,"--total-steps",str(args.steps),"--warmup-steps",str(args.warmup),"--vocab-size",str(args.vocab_size),"--ffn-type",args.ffn_type,"--evaluation-interval",str(args.evaluation_interval),"--evaluation-batches",str(args.evaluation_batches),"--checkpoint-interval",str(args.checkpoint_interval),"--seed",str(args.seed),"--device","cuda",]

script_args.append("--use-rmsnorm" if args.rmsnorm else "--no-use-rmsnorm")
script_args.append("--use-rope" if args.rope else "--no-use-rope")
script_args.append("--use-bf16" if args.bf16 else "--no-use-bf16")

client = AfriLinkClient()
client.authenticate(api_key=os.environ["AFRILINK_API_KEY"])

job = client.train(script="afrilink_entry.py",container="afrilink-finetune",data=args.data,gpus=1,time_limit=args.time_limit,script_args=script_args,)

print(f"Submitting {args.name}: " f"lr={args.lr}, steps={args.steps}, " f"vocab={args.vocab_size}, " f"rmsnorm={args.rmsnorm}, rope={args.rope}, " f"ffn={args.ffn_type}, bf16={args.bf16}")

result = job.run(wait=True)

print(result)
print(job.get_logs(tail=100))

if result["status"] != "completed":
    raise RuntimeError(f"{args.name} ended with status {result['status']}")

destination = Path("afrilink_results") / args.group / args.name
destination.mkdir(parents=True, exist_ok=True)

client.download_model(result["job_id"],str(destination),)

print(f"Downloaded results to {destination}")