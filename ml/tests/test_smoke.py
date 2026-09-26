"""End-to-end smoke test: fake CFD profiles + fake digitized DNS figures ->
tiny training run -> evaluate. Confirms the pipeline runs; it says nothing
about accuracy (that needs real CFD data and a full training run).

Run: cd ml && python3 -m pytest tests/test_smoke.py -q
"""
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

ML_DIR = Path(__file__).resolve().parent.parent
REPO = ML_DIR.parent
R, A, NU = 0.07, 0.0775, 0.0712 / 9800
PRS = (0.025, 1.0, 2.0, 7.0)


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


def fake_cfd_field(folder):
    """Coarse fake 'mesh': cells on a polar grid, plus rod-surface rows."""
    rows = []
    add = lambda *r: rows.append(r)
    for i in range(12):
        phi = (i + 0.5) / 12 * math.pi / 2
        rho_max = A / max(math.cos(phi), math.sin(phi))
        for j in range(10):
            d = (rho_max - R) * ((j + 0.5) / 10) ** 2
            x, y = (R + d) * math.cos(phi), (R + d) * math.sin(phi)
            f = 1 - math.exp(-d / 0.002)
            add("flow", "cell", x, y, 1e-5, "w", 1.2 * f, 9800, "", "")
            add("flow", "cell", x, y, 1e-5, "nut", 50 * NU * f**2, 9800, "", "")
            add("flow", "cell", x, y, 1e-5, "k", 0.003 * f**2, 9800, "", "")
            for pr in PRS:
                add(f"Pr{pr:g}_isoT", "cell", x, y, 1e-5, "T", -5 * pr**0.3 * f, 9800, pr, "isoT")
                add(f"Pr{pr:g}_isoFlux", "cell", x, y, 1e-5, "T", -500 * pr**0.3 * f, 9800, pr, "isoFlux")
        x, y = R * math.cos(phi), R * math.sin(phi)
        add("flow", "wall", x, y, 0.0, "tau_w", 1.2 * NU / 0.002, 9800, "", "")
        for pr in PRS:
            add(f"Pr{pr:g}_isoT", "wall", x, y, 0.0, "q_w", 0.02, 9800, pr, "isoT")
            add(f"Pr{pr:g}_isoFlux", "wall", x, y, 0.0, "T", 0.0, 9800, pr, "isoFlux")
    cols = ["case_id", "kind", "x", "y", "area", "quantity", "value", "Re", "Pr", "bc"]
    pd.DataFrame(rows, columns=cols).to_csv(folder / "cfd_field.csv", index=False)
    pd.DataFrame({"Pr": [p for p in PRS for _ in (0, 1)], "bc": ["isoT", "isoFlux"] * 4,
                  "Nu_CFD": [10.0] * 8, "wall_heat_balance_error": [0.0] * 8}).to_csv(
        folder / "cfd_nusselt.csv", index=False)


def fake_dns_digitized(folder):
    pd.DataFrame({"case_id": ["dns"] * 3, "x": [0, 20, 45], "y": [0.6, 1.0, 1.1]}).to_csv(
        folder / "dns_fig6_wall_shear.csv", index=False)
    pd.DataFrame({"case_id": [45] * 3, "x": [1, 10, 100], "y": [1, 10, 17]}).to_csv(
        folder / "dns_fig7_velocity_wall_units.csv", index=False)
    pd.DataFrame({"case_id": [1.0] * 3, "x": [0, 20, 45], "y": [0.5, 1.0, 1.2]}).to_csv(
        folder / "dns_fig11a_wall_heat_flux_isoT.csv", index=False)
    pd.DataFrame({"case_id": [1.0] * 3, "x": [0.05, 0.8, 1.6], "y": [0.3, 1.2, 0.5]}).to_csv(
        folder / "dns_fig12a_temperature_isoT.csv", index=False)
    pd.DataFrame({"case_id": ["uu", "vv", "ww"] * 3, "x": [0.05] * 3 + [0.6] * 3 + [1.2] * 3,
                  "y": [1, 0.5, 3, 1.2, 0.8, 2.7, 0.6, 0.6, 0.8]}).to_csv(
        folder / "dns_fig9a_normal_stresses.csv", index=False)


def test_train_and_evaluate_pipeline():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        digitized = tmp / "digitized_data"
        fake_cfd_profiles(digitized / "cfd_generated" / "cfd_profiles.csv")
        fake_cfd_field(digitized / "cfd_generated")
        fake_dns_digitized(digitized)

        cfg = yaml.safe_load((ML_DIR / "configs" / "default.yaml").read_text())
        cfg["training"].update(epochs_data=2, epochs_physics=2, ramp_epochs=1, n_collocation=64,
                               n_boundary=16, batch_data=256, print_every=1)
        cfg["model"].update(momentum_hidden=[32, 32], thermal_hidden=[32, 32], fourier_features=8)
        cfg["paths"].update(
            cfd_field=str(digitized / "cfd_generated" / "cfd_field.csv"),
            cfd_nusselt=str(digitized / "cfd_generated" / "cfd_nusselt.csv"),
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
                         "dns_fig9a_tke.png", "dns_fig11a_wall_heat_flux.png", "dns_fig12a_temperature.png"):
            assert expected in plots, f"missing {expected}"

        # sparse-data experiment helper: thinned data, lines only, and the plain-NN baseline
        sys.path.insert(0, str(ML_DIR / "src"))
        import experiments
        runs = {"half PINN": {"data_fraction": 0.5}, "plain": {"use_physics": False},
                "lines": {"data_source": "lines"}}
        table = experiments.run_all(cfg, runs, out_dir=str(tmp / "exp"))
        assert list(table["run"]) == list(runs)

        # explainability on the trained model, and sensor importance from line-dropping runs
        import matplotlib
        matplotlib.use("Agg")
        import xai
        model = xai.load_trained(cfg)
        assert len(xai.residual_maps(model, cfg)) == 5
        assert len(xai.energy_budget(model, cfg)) == len(cfg["thermal"]["pr_values"])
        sensor = {"lines PINN": {"data_source": "lines"},
                  "lines PINN without seg1": {"data_source": "lines", "data_lines": ["seg2", "seg3", "line15"]}}
        experiments.run_all(cfg, sensor, out_dir=str(tmp / "exp"))
        results = [json.loads(p.read_text()) for p in (tmp / "exp").glob("*/result.json")]
        assert len(xai.sensor_importance(results)) == 1


if __name__ == "__main__":
    test_train_and_evaluate_pipeline()
    print("smoke test passed")
