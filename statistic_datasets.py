"""
Analyze case distribution in MJSynth dataset
"""
import os
from pathlib import Path
import matplotlib.pyplot as plt

# CACHE_DIR = Path("/home/hieu/.cache/huggingface/datasets/c7d0c699152e5a310ad6b793bba5e302f28699ba")
CACHE_DIR = Path("D:/huggingface_cache")
CACHE_DIR.mkdir(exist_ok=True)
os.environ['HF_HOME'] = str(CACHE_DIR)
os.environ['HF_DATASETS_CACHE'] = str(CACHE_DIR / 'datasets')

from datasets import load_dataset
from collections import Counter, defaultdict
from typing import Dict




def classifyLabel(s: str) -> str:
    """
    Classify a text label by its case pattern
    
    Args:
        s: Input string to classify
        
    Returns:
        'all_upper': All letters are uppercase (e.g., "HELLO", "HELLO123")
        'all_lower': All letters are lowercase (e.g., "hello", "hello123")
        'mixed': Mix of upper and lowercase (e.g., "Hello", "HeLLo")
        'no_letters': No letters found (e.g., "123", "!@#")
    """
    letters = [c for c in s if c.isalpha()]
    
    if not letters:
        return "no_letters"
    
    if all(c.isupper() for c in letters):
        return "all_upper"
    
    if all(c.islower() for c in letters):
        return "all_lower"
    
    return "mixed"


def computeStatistics(ds, max_samples=None) -> Dict:
    """
    Compute case distribution statistics for a dataset
    
    Args:
        ds: HuggingFace dataset
        max_samples: Maximum number of samples to analyze (None = all)
        
    Returns:
        Dictionary with total, stats, and examples
    """
    counts = Counter()
    examples = defaultdict(list)
    total = 0
    
    # Determine how many samples to process
    num_samples = len(ds) if max_samples is None else min(max_samples, len(ds))
    
    print(f"\nAnalyzing {num_samples:,} samples...")
    print("-" * 70)
    
    for i in range(num_samples):
        # FIX 2: Simplified access
        label = ds[i]['label']
        
        # Show progress
        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i+1:,}/{num_samples:,} ({(i+1)/num_samples*100:.1f}%)")
        
        total += 1
        cat = classifyLabel(label)
        counts[cat] += 1
        
        # Store examples (limit to 10 per category)
        if len(examples[cat]) < 10:
            examples[cat].append(label)
    
    # Calculate percentages
    stats = {
        k: {
            "count": v, 
            "percent": (v / total * 100 if total else 0.0)
        } 
        for k, v in counts.items()
    }
    
    return {
        "total": total, 
        "stats": stats, 
        "examples": dict(examples)
    }

# flow chart to statistics case distribution as image
def export_case_distribution_chart(stats: Dict, output_path="./imgs/case_distribution.png"):
    """
    Export a bar chart with single-row annotations: 'count (percent%)'
    """
    from pathlib import Path
    import matplotlib.pyplot as plt

    labels = list(stats.keys())
    counts = [stats[k]["count"] for k in labels]
    percents = [stats[k]["percent"] for k in labels]

    display_labels = [(lbl.replace('_', ' ').title() if lbl else '') for lbl in labels]

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(labels))]

    bars = ax.bar(range(len(labels)), counts, color=colors, edgecolor='black', linewidth=0.7)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(display_labels, ha='right', fontsize=10)
    ax.set_ylabel("Number of Samples")
    ax.set_title("Case Distribution in MJSynth Dataset", fontsize=14, pad=12)

    max_count = max(counts) if counts else 1
    # Annotate as single row: "count (percent%)"
    for rect, cnt, pct in zip(bars, counts, percents):
        x = rect.get_x() + rect.get_width() / 2
        y = rect.get_height()
        ax.text(x, y + max_count * 0.03, f"{cnt:,} ({pct:.1f}%)", ha="center", va="bottom", fontsize=9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.0)

    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"📁 Chart saved to: {output_path}")



def printStatistics(result: Dict):
    """Pretty print statistics"""
    print("\n" + "=" * 70)
    print("CASE DISTRIBUTION STATISTICS")
    print("=" * 70)
    print(f"Total samples analyzed: {result['total']:,}")
    print()
    
    # Sort by count (descending)
    sorted_stats = sorted(
        result['stats'].items(), 
        key=lambda x: x[1]['count'], 
        reverse=True
    )
    
    for category, data in sorted_stats:
        count = data['count']
        percent = data['percent']
        print(f"{category:12s}: {count:7,} samples ({percent:5.2f}%)")
    
    print("\n" + "=" * 70)
    print("SAMPLE EXAMPLES (up to 10 per category)")
    print("=" * 70)
    
    for category, labels in result['examples'].items():
        print(f"\n{category}:")
        for i, label in enumerate(labels[:10], 1):
            print(f"  {i:2d}. '{label}'")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    stats = result['stats']
    export_case_distribution_chart(result_quick["stats"])
    if 'all_upper' in stats:
        upper_percent = stats['all_upper']['percent']
        if upper_percent > 70:
            print("⚠️  DATASET IS HEAVILY BIASED TOWARDS UPPERCASE!")
            print(f"   {upper_percent:.1f}% of samples are all uppercase.")
            print("   This explains why your model outputs uppercase text.")
            print("\n💡 Solution:")
            print("   - Convert all labels to lowercase during training")
            print("   - Or use data augmentation to balance case distribution")
        elif upper_percent > 50:
            print("⚠️  Dataset has MORE uppercase samples")
            print(f"   {upper_percent:.1f}% are uppercase vs {stats.get('all_lower', {}).get('percent', 0):.1f}% lowercase")
        else:
            print("✓  Dataset appears relatively balanced")
    
    print("=" * 70)

def testClassifier():
    """Test the classifyLabel function"""
    print("\n" + "=" * 70)
    print("TESTING CLASSIFIER FUNCTION")
    print("=" * 70)
    
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
        ("UPPER-CASE", "all_upper"),
        ("lower_case", "all_lower"),
        ("CamelCase", "mixed"),
    ]
    
    passed = 0
    failed = 0
    
    for i, (label, expected) in enumerate(test_cases, 1):
        actual = classifyLabel(label)
        ok = actual == expected
        status = "✓ PASS" if ok else "✗ FAIL"
        
        if ok:
            passed += 1
        else:
            failed += 1
        
        print(f"{i:2d}. {label!r:20} expected={expected:10} actual={actual:10} {status}")
    
    print(f"\nResults: {passed} passed, {failed} failed")
    print("=" * 70)


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("MJSYNTH DATASET CASE ANALYSIS")
    print("=" * 70)
    
    # Step 1: Test the classifier
    testClassifier()
    
    # Step 2: Load dataset
    print("\n[1/3] Loading dataset...")
    try:
        # FIX 3: Load from HuggingFace directly (recommended)
        ds = load_dataset("priyank-m/MJSynth_text_recognition", split="train")
        print(f"✓ Loaded {len(ds):,} samples from HuggingFace")
    except Exception as e:
        print(f"✗ Error loading from HuggingFace: {e}")
        print("\nTrying to load from local cache...")
        try:
            # Alternative: Load from local cache
            cache_path = CACHE_DIR / "datasets" / "priyank-m___mj_synth_text_recognition"
            ds = load_dataset(str(cache_path), split="train")
            print(f"✓ Loaded {len(ds):,} samples from cache")
        except Exception as e2:
            print(f"✗ Error loading from cache: {e2}")
            print("\nPlease ensure dataset is downloaded first.")
            exit(1)
    
    print(f"Dataset keys: {list(ds[0].keys())}")
    print(f"First example: {ds[0]}")
    
    # Step 3: Analyze case distribution
    print("\n[2/3] Computing statistics...")
    
    # Option A: Quick analysis (first 10,000 samples)
    print("\n📊 Quick Analysis (10,000 samples):")
    result_quick = computeStatistics(ds, max_samples=10000)
    printStatistics(result_quick)
    
    # Option B: Full analysis (uncomment to run - takes longer)
    # print("\n\n📊 Full Analysis (ALL samples):")
    # result_full = computeStatistics(ds, max_samples=None)
    # printStatistics(result_full)
    
    # Step 4: Summary
    print("\n[3/3] Analysis complete!")


