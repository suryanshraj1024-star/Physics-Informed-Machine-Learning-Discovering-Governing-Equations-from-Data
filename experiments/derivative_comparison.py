import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments import derivative_experiment, plot_derivative_comparison

rows = derivative_experiment()
for row in rows:
    print(row)
plot_derivative_comparison(rows)
