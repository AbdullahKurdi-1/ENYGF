"""Compare the CFD (not the PINN) with the 2023 DNS at the key points the
papers discuss: wall shear in the gap and at 45 deg, turbulent kinetic
energy in the gap / mid-path / subchannel centre, excess temperature at the
gap and subchannel centres, mean friction velocity.

DNS values come from the digitized figures (digitized_data/dns_*.csv, made
by digitized_data/digitize_dns_figures.py). The 2018 URANS study (same
k-omega SST model) gives a narrow-gap-centre velocity of 0.55 U_b (its
Fig. 11) - the same as this CFD - which shows the gap deficit is the model's,
not this set-up's.

For the full comparison (curves, CFD and PINN, RMSE per figure) use
validation.validate(); this script prints only the headline numbers.

Usage (from ml/):  python3 src/compare_cfd_to_papers.py
Notebook:          import compare_cfd_to_papers; compare_cfd_to_papers.main(cfg)
"""
import argparse
from pathlib import Path

import yaml

import validation


def main(cfg):
    return validation.key_points(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    main(yaml.safe_load(Path(parser.parse_args().config).read_text()))
