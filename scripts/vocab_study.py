
"""
Choosing a vocabulary size
Train BPE at a range of vocabulary
sizes (at least 1,000 / 2,000 / 4,000 / 8,000 / 16,000) on the validation file or a subsample of the training
file, and measure the compression ratio for each.
Note that you can track the corpus token count incrementally during training almost for free — every
merge reduces it by exactly the count of the merged pair.

must produce curve and probably a csv 

"""

from __future__ import annotations
import csv 
import sys 
from pathlib import Path