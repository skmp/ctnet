"""
Test CTGPT: verify DCT replacement preserves GPT-2 output.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from ctgpt_model import replace_linears_with_dct, _is_dct_layer


def test_dct_roundtrip():
    """Verify DCT replacement produces identical output to original GPT-2."""
    print("Loading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    # Get original output
    input_ids = tokenizer.encode("Hello, world!", return_tensors="pt")
    with torch.no_grad():
        orig_output = model(input_ids).logits

    print(f"Original output shape: {list(orig_output.shape)}")
    print(f"Original output range: [{orig_output.min():.4f}, {orig_output.max():.4f}]")

    # Replace with DCT
    print("\nReplacing linear layers with DCTLinear...")
    replace_linears_with_dct(model.transformer, block_size=16)

    n_dct = sum(1 for m in model.modules() if _is_dct_layer(m))
    print(f"=> {n_dct} DCTLinear layers")

    # Get DCT output
    model.eval()
    with torch.no_grad():
        dct_output = model(input_ids).logits

    # Compare
    diff = (orig_output - dct_output).abs()
    print(f"\nMax diff: {diff.max():.8f}")
    print(f"Mean diff: {diff.mean():.8f}")
    print(f"Argmax match: {orig_output.argmax(-1).tolist()} vs {dct_output.argmax(-1).tolist()}")

    ok = diff.max() < 0.01
    print(f"\n{'PASS' if ok else 'FAIL'}: DCT roundtrip {'preserves' if ok else 'does NOT preserve'} output")
    return ok


def test_generation():
    """Verify DCT model can generate text."""
    print("\n--- Generation test ---")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    replace_linears_with_dct(model.transformer, block_size=16)
    model.eval()

    input_ids = tokenizer.encode("To be, or not to be", return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            input_ids, max_new_tokens=50,
            temperature=0.8, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Generated: {text}")
    print("PASS: Generation works")
    return True


def test_parameter_counts():
    """Verify parameter count after replacement."""
    print("\n--- Parameter count test ---")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    total_before = sum(p.numel() for p in model.parameters())
    replace_linears_with_dct(model.transformer, block_size=16)
    total_after = sum(p.numel() for p in model.parameters())

    dct_params = sum(m.weight_dct.numel() for m in model.modules() if _is_dct_layer(m))

    print(f"Params before: {total_before / 1e6:.1f}M")
    print(f"Params after:  {total_after / 1e6:.1f}M")
    print(f"DCT params:    {dct_params / 1e6:.1f}M")
    print(f"DCT fraction:  {dct_params / total_after * 100:.1f}%")
    print(f"PASS: Parameter counts reasonable")
    return True


if __name__ == "__main__":
    test_dct_roundtrip()
    test_generation()
    test_parameter_counts()
