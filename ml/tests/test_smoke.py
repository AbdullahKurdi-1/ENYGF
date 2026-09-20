"""End-to-end smoke test: dataset loading -> tiny training run -> evaluate.
Not a correctness test of the physics (that needs real CFD/digitized data) -
just confirms the pipeline runs without crashing after any code change.

Run with: cd ml && python3 -m pytest tests/test_smoke.py -q
(or: python3 tests/test_smoke.py, it also runs standalone)
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ML_DIR = Path(__file__).resolve().parent.parent

FAKE_DIGITIZED = {
    "velocity_line1_by_Re.csv": "case_id,x,y\nR5,0.0,0.0\nR5,1.0,1.0\n",
    "tke_line1_by_Re.csv": "case_id,x,y\nR5,0.0,0.0\nR5,1.0,0.02\n",
    "temperature_line1_constT_by_Pr.csv": "case_id,x,y\n1,0.0,0.0\n1,1.0,1.0\n",
    "temperature_line2_constT_by_Pr.csv": "case_id,x,y\n",
    "temperature_line1_constq_by_Pr.csv": "case_id,x,y\n",
    "temperature_line2_constq_by_Pr.csv": "case_id,x,y\n",
}


def test_train_and_evaluate_pipeline():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data_dir = tmp / "digitized_data"
        data_dir.mkdir()
        for name, content in FAKE_DIGITIZED.items():
            (data_dir / name).write_text(content)

        cfg = yaml.safe_load((ML_DIR / "configs" / "default.yaml").read_text())
        cfg["training"]["epochs"] = 3
        cfg["training"]["n_collocation"] = 32
        cfg["paths"]["digitized_data"] = str(data_dir)
        cfg["paths"]["checkpoint"] = str(tmp / "ckpt.pt")
        cfg["paths"]["plots_dir"] = str(tmp / "plots")
        cfg["paths"]["table2"] = str(ML_DIR.parent / "paper_reference" / "table2_reynolds_scaling.csv")
        cfg_path = tmp / "config.yaml"
        yaml.safe_dump(cfg, cfg_path.open("w"))

        subprocess.run(
            [sys.executable, "src/train.py", "--config", str(cfg_path)],
            check=True, cwd=ML_DIR,
        )
        assert (tmp / "ckpt.pt").exists()

        subprocess.run(
            [sys.executable, "src/evaluate.py", "--config", str(cfg_path)],
            check=True, cwd=ML_DIR,
        )
        assert any((tmp / "plots").glob("*.png"))


if __name__ == "__main__":
    test_train_and_evaluate_pipeline()
    print("smoke test passed")
