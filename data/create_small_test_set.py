import pandas as pd
from pathlib import Path

def sample_small_test_set(file, n_rows):
    df = pd.read_csv(file)
    df= df.sample(n_rows, random_state=42)
    out_path = Path("data") / "test" / f"small_test_set_machine.csv"
    df.to_csv(out_path, index=False)




in_path = Path("data") / "test" / f"test_dataset.csv"
in_p2 = Path("data") / "test" / f"test_machine_data.csv"
sample_small_test_set(in_p2, 1000)