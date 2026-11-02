#!/usr/bin/env python3
"""One-time export (the only Python in the project).

Writes:
  <out>/weights.bin  flat fp16 TinyLlama weights with a small header
  <out>/refs.bin     reference prompts (token IDs) + HF greedy continuations + per-step top-k logits

weights.bin layout (little endian):
  header (64 bytes): char[4] "KVPW", int32 version=1, int32 vocab, hidden, intermediate, layers,
                     heads, kv_heads, head_dim, max_pos, float32 rope_theta, rms_eps, pad
  embed       [vocab, hidden]
  per layer:  attn_norm [hidden]
              wqkv      [hidden + 2*kv_heads*head_dim, hidden]   (q, k, v rows concatenated)
              wo        [hidden, hidden]
              ffn_norm  [hidden]
              w_gate_up [2*intermediate, hidden]                  (gate rows, then up rows)
              w_down    [hidden, intermediate]
  final_norm  [hidden]
  lm_head     [vocab, hidden]
All matrices are row-major [out_features, in_features] (PyTorch nn.Linear layout), fp16.

refs.bin layout:
  char[4] "KVPR", int32 num_prompts, int32 num_steps, int32 topk
  per prompt: int32 len, int32 tokens[len], int32 greedy[num_steps],
              int32 top_ids[num_steps*topk], float32 top_logits[num_steps*topk]
Reference logits come from an fp32 HF forward pass with greedy decoding that ignores EOS.

Usage: python tools/export_weights.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --out data
"""
import argparse
import os
import struct

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = ("You are a helpful, concise assistant. Answer the user's question accurately and "
          "explain your reasoning step by step when it helps.")
PROMPTS = [
    "What is the capital of France, and why is it historically important?",
    "Write a short poem about the ocean at night.",
    "Explain how a hash table works to a ten year old.",
    "List three advantages of electric cars over petrol cars.",
    "What is the difference between a process and a thread in an operating system?",
    "Give me a recipe for a simple tomato pasta sauce.",
    "Summarize the plot of Romeo and Juliet in a few sentences.",
    "How does paging work in virtual memory, and why do operating systems use it?",
]


def write_tensor(f, t):
    a = t.detach().to(torch.float16).contiguous().numpy()
    f.write(a.tobytes())
    return a.nbytes


def export_weights(model, path):
    cfg = model.config
    H, I, L = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers
    nh, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = H // nh
    sd = model.state_dict()
    with open(path, "wb") as f:
        hdr = struct.pack("<4s9i2f", b"KVPW", 1, cfg.vocab_size, H, I, L, nh, nkv, hd,
                          cfg.max_position_embeddings, float(getattr(cfg, "rope_theta", 10000.0)),
                          float(cfg.rms_norm_eps))
        f.write(hdr.ljust(64, b"\0"))
        n = write_tensor(f, sd["model.embed_tokens.weight"])
        for l in range(L):
            p = f"model.layers.{l}."
            n += write_tensor(f, sd[p + "input_layernorm.weight"])
            n += write_tensor(f, torch.cat([sd[p + "self_attn.q_proj.weight"],
                                            sd[p + "self_attn.k_proj.weight"],
                                            sd[p + "self_attn.v_proj.weight"]], 0))
            n += write_tensor(f, sd[p + "self_attn.o_proj.weight"])
            n += write_tensor(f, sd[p + "post_attention_layernorm.weight"])
            n += write_tensor(f, torch.cat([sd[p + "mlp.gate_proj.weight"],
                                            sd[p + "mlp.up_proj.weight"]], 0))
            n += write_tensor(f, sd[p + "mlp.down_proj.weight"])
        n += write_tensor(f, sd["model.norm.weight"])
        lm = sd.get("lm_head.weight", sd["model.embed_tokens.weight"])
        n += write_tensor(f, lm)
    print(f"wrote {path}: {n / 2**30:.2f} GiB of fp16 tensors "
          f"(H={H} I={I} L={L} heads={nh} kv_heads={nkv} head_dim={hd})")


@torch.no_grad()
def greedy_reference(model, ids, steps, topk):
    """Greedy decode ignoring EOS. Returns tokens, top-k ids and logits for each step."""
    out = model(torch.tensor([ids]), use_cache=True)
    past = out.past_key_values
    logits = out.logits[0, -1].float()
    toks, tids, tvals = [], [], []
    for _ in range(steps):
        v, i = torch.topk(logits, topk)
        tids.append(i.tolist())
        tvals.append(v.tolist())
        nxt = int(i[0])
        toks.append(nxt)
        out = model(torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1].float()
    return toks, tids, tvals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--out", default="data")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--skip-weights", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()

    if not args.skip_weights:
        export_weights(model, os.path.join(args.out, "weights.bin"))

    path = os.path.join(args.out, "refs.bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<4s3i", b"KVPR", len(PROMPTS), args.steps, args.topk))
        for k, p in enumerate(PROMPTS):
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
            if hasattr(ids, "input_ids"):
                ids = ids["input_ids"]
            ids = [int(x) for x in ids]
            toks, tids, tvals = greedy_reference(model, ids, args.steps, args.topk)
            f.write(struct.pack("<i", len(ids)))
            f.write(np.asarray(ids, np.int32).tobytes())
            f.write(np.asarray(toks, np.int32).tobytes())
            f.write(np.asarray(tids, np.int32).tobytes())
            f.write(np.asarray(tvals, np.float32).tobytes())
            print(f"prompt {k}: {len(ids)} tokens -> {tok.decode(toks)[:70]!r}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
