from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from utils import tokenize

TEXT_PATH = "src/data/text8"
SAVE_DIR = "analysis"

# Load text8 (downloaded separately)
text8_path = Path(TEXT_PATH)
print("Loading corpus...")
text = text8_path.read_text()
tokens = tokenize(text)

# Count word frequencies
counts = Counter(tokens)
# Sort by frequency (highest first)
freqs = np.array(sorted(counts.values(), reverse=True))

# Make rank axis
ranks = np.arange(1, len(freqs) + 1)

# Plot log-log distribution
plt.figure(figsize=(8,5))
plt.loglog(ranks, freqs, marker=".")
plt.xlabel("Rank of word (1 = most frequent)")
plt.ylabel("Frequency (count)")
plt.title("Word frequency distribution in text8 (Zipf's law)")
plt.grid(True, which="both", ls="--", lw=0.5)

# Add infobox with statistics
vocab_size = len(freqs)
most_frequent_count = freqs[0]
median_freq = np.median(freqs)

infobox_text = f"Vocabulary size: {vocab_size:,}\nMost frequent word count: {most_frequent_count:,}\nMedian frequency: {median_freq}"

# Position the infobox in the upper right corner
plt.text(0.98, 0.98, infobox_text, 
         transform=plt.gca().transAxes,
         verticalalignment='top',
         horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
         fontsize=10,
         fontfamily='monospace')

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/word_freq_dist.png", dpi=300)