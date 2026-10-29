"""Evaluate frozen Base with matched latent draws and two initial velocities."""
import argparse
import sys
from reproduction_base_noise_probe import main


if __name__ == '__main__':
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--velocity_audit', required=True,
                   help='Path to a newly generated 50-episode velocity audit; original artifacts were archived away')
    args, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    main(probe_kind='reset_velocity', velocity_audit=args.velocity_audit)
