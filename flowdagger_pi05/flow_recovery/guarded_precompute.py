"""Wait for GPU load headroom after slow imports, then run the unchanged cache builder."""
import argparse
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--min-free-mib', type=int, required=True)
    p.add_argument('--wait-marker', type=Path, help='Finish predecessor before loading model weights')
    args = p.parse_args()
    # Import latency on shared storage can outlast the scheduler's resource snapshot.
    from flowdagger_pi05.libero_offline.bridge import LiberoBridge  # noqa: F401
    while args.wait_marker and not args.wait_marker.is_file():
        print(f'Waiting for predecessor completion: {args.wait_marker}', flush=True)
        time.sleep(30)
    gpu = int(os.environ['CUDA_VISIBLE_DEVICES'])
    while True:
        free = int(subprocess.check_output(
            ['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
        print(f'Checkpoint load guard: gpu={gpu} free={free} MiB required={args.min_free_mib} MiB', flush=True)
        if free >= args.min_free_mib:
            break
        time.sleep(30)
    sys.argv = ['precompute', '--config', args.config]
    runpy.run_module('flowdagger_pi05.libero_offline.precompute', run_name='__main__')


if __name__ == '__main__':
    main()
