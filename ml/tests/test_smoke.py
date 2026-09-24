"""End-to-end smoke test: fake CFD profiles + fake digitized DNS figures ->
tiny training run -> evaluate. Confirms the pipeline runs; it says nothing
about accuracy (that needs real CFD data and a full training run).

Run: cd ml && python3 -m pytest tests/test_smoke.py -q
"""
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

ML_DIR = Path(__file__).resolve().parent.parent
REPO = ML_DIR.parent
R, A, NU = 0.07, 0.0775, 8.0099e-6


def fake_cfd_profiles(path):
    rows = []
    for s_i in range(20):
        d = (A - R) * s_i / 19
        x, y = R + d, 1e-6
        rows.append(("flow", "seg1", d, d, x, y, "w", 1 - math.exp(-d / 0.002), 9800, "", ""))
        rows.append(("flow", "seg1", d, d, x, y, "nut", 50 * NU * (1 - math.exp(-d / 0.002)) ** 2, 9800, "", ""))
        rows.append(("flow", "seg1", d, d, x, y, "k", 0.003 * (1 - math.exp(-d / 0.002)) ** 2, 9800, "", ""))
        for pr, bc in ((1.0, "isoT"), (0.025, "isoFlux")):
            rows.append((f"Pr{pr:g}_{bc}", "seg1", d, d, x, y, "T", -5 * (1 - math.exp(-d / 0.003)), 9800, pr, bc))
    cols = ["case_id", "line", "s", "xi", "x", "y", "quantity", "value", "Re", "Pr", "bc"]
    path.parent.mkdir(parents=True)
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def fake_dns_digitized(folder):
    pd.DataFrame({"case_id": ["dns"] * 3, "x": [0, 20, 45], "y": [0.6, 1.0, 1.1]}).to_csv(
        folder / "dns_fig6_wall_shear.csv", index=False)
    pd.DataFrame({"case_id": [45] * 3, "x": [1, 10, 100], "y": [1, 10, 17]}).to_csv(
        folder / "dns_fig7_velocity_wall_units.csv", index=False)
    pd.DataFrame({"case_id": [1.0] * 3, "x": [0, 20, 45], "y": [0.5, 1.0, 1.2]}).to_csv(
        folder / "dns_fig11a_wall_heat_flux_isoT.csv", index=False)
    pd.DataFrame({"case_id": [1.0] * 3, "x": [0.05, 0.8, 1.6], "y": [0.3, 1.2, 0.5]}).to_csv(
        folder / "dns_fig12a_temperature_isoT.csv", index=False)


def test_train_and_evaluate_pipeline():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        digitized = tmp / "digitized_data"
        fake_cfd_profiles(digitized / "cfd_generated" / "cfd_profiles.csv")
        fake_dns_digitized(digitized)

        cfg = yaml.safe_load((ML_DIR / "configs" / "default.yaml").read_text())
        cfg["training"].update(epochs_momentum=2, epochs_joint=2, n_collocation=64, n_boundary=16, print_every=1)
        cfg["model"].update(momentum_hidden=[32, 32], thermal_hidden=[32, 32], fourier_features=8)
        cfg["paths"].update(
            cfd_profiles=str(digitized / "cfd_generated" / "cfd_profiles.csv"),
            digitized_dir=str(digitized),
            dns_nusselt=str(REPO / "paper_reference" / "table_dns_2023_nusselt.csv"),
            checkpoint=str(tmp / "out" / "ckpt.pt"),
            loss_history=str(tmp / "out" / "loss_history.csv"),
            plots_dir=str(tmp / "out" / "plots"),
        )
        cfg_path = tmp / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))

        subprocess.run([sys.executable, "src/train.py", "--config", str(cfg_path)], check=True, cwd=ML_DIR)
        assert (tmp / "out" / "ckpt.pt").exists()
        assert len(pd.read_csv(tmp / "out" / "loss_history.csv")) == 4

        subprocess.run([sys.executable, "src/evaluate.py", "--config", str(cfg_path)], check=True, cwd=ML_DIR)
        plots = {p.name for p in (tmp / "out" / "plots").glob("*.png")}
        for expected in ("cfd_fit_flow.png", "nusselt_vs_dns.png", "dns_fig6_wall_shear.png",
                         "dns_fig7_velocity_wall_units.png", "dns_fig11a_wall_heat_flux.png",
                         "dns_fig12a_temperature.png"):
            assert expected in plots, f"missing {expected}"


if __name__ == "__main__":
    test_train_and_evaluate_pipeline()
    print("smoke test passed")
