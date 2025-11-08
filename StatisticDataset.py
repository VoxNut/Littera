from collections import Counter, defaultdict
from datasets import load_dataset
from typing import Any, Dict, Optional
# print("Loading 1,000,000 samples from MJSynth dataset...")
# ds = load_dataset("priyank-m/MJSynth_text_recognition", split="train[:20]")
# print(f"Loaded {len(ds)} samples successfully!")

path = "/home/hieu/.cache/huggingface/datasets/priyank-m___mj_synth_text_recognition/default/0.0.0/c7d0c699152e5a310ad6b793bba5e302f28699ba"

# assume `path` variable is already defined and points to the local dataset directory
ds = load_dataset(path, split="train")

print("Total samples:", len(ds))
print("Example keys:", list(ds[0].keys()))
print("First example:", ds[0])

def detectTextField(sample: dict) -> str:
    for k, v in sample.items():
        if isinstance(v, str):
            return k
    raise ValueError("No string field found in dataset example to use as label/text.")

def classifyLabel(s: str) -> str:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return "no_letters"
    if all(c.isupper() for c in letters):
        return "all_upper"
    if all(c.islower() for c in letters):
        return "all_lower"
    return "mixed"

def getDs(ds_or_path: Any, split: str = "train"):
    ds = load_dataset(ds_or_path, split=split)
    return ds

def computeStatistics(
    ds
) -> Dict[str, object]:
    maxExample = 20
    counts = Counter()
    examples = defaultdict(list)
    total = 0
    textField = ''
    for i, ex in enumerate(ds):
        textField = ds[i]['label']

        if maxExample is not None and i >= maxExample:
            break
        print(textField)
        total += 1
        cat = classifyLabel(textField)
        counts[cat] += 1
        examples[cat].append(textField)


    stats = {k: {"count": v, "percent": (v / total * 100 if total else 0.0)} for k, v in counts.items()}
    print('example: ', examples)
    return {"total": total, "stats": stats, "examples": dict(examples)}

# python
test_cases = [
    ("HELLO", "all_upper"),
    ("hello", "all_lower"),
    ("Hello", "mixed"),
    ("HELLO123", "all_upper"),
    ("hello123", "all_lower"),
    ("HeLLo123", "mixed"),
    ("", "no_letters"),
    ("12345", "no_letters"),
    ("!@#$", "no_letters"),
    ("UPPER-CASE", "all_upper"),   # hyphen ignored, letters are upper
    ("lower_case", "all_lower"),   # underscore ignored, letters are lower
    ("CamelCase", "mixed"),     # emoji ignored, letters 'emoji' are lower
]


for i, (label, expected) in enumerate(test_cases, 1):
    actual = classifyLabel(label)
    ok = actual == expected
    print(f"{i:2d}. {label!r:20} expected={expected:10} actual={actual:10} {'PASS' if ok else 'FAIL'}")

print(computeStatistics(getDs(path)).get('stats'))