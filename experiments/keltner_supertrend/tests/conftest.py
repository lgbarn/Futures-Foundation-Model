"""Put the repo root on sys.path so `experiments.keltner_supertrend.*` imports."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
