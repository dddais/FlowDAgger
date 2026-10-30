"""Compile the actual service once before measured trials, without websocket ping deadlines."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from openpi_client import msgpack_numpy
from websockets.sync.client import connect


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--task', type=int, required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    pool = json.loads((Path(config['dataset_root']) / 'data_pool.json').read_text())
    task = next(t for t in pool['tasks'] if t['task_index'] == args.task)
    request = {'task_key': task['task_key'], 'prompt': task['official_task_language'], 'noise_seed': 0,
               'observation/image': np.zeros((256, 256, 3), np.uint8),
               'observation/wrist_image': np.zeros((256, 256, 3), np.uint8),
               'observation/state': np.zeros(8, np.float32)}
    started = time.monotonic()
    with connect(f'ws://127.0.0.1:{args.port}', proxy=None, ping_interval=None,
                 compression=None, max_size=None) as ws:
        metadata = msgpack_numpy.unpackb(ws.recv())
        ws.send(msgpack_numpy.Packer().pack(request))
        response = ws.recv(timeout=1200)
        if isinstance(response, str):
            raise RuntimeError(response)
        response = msgpack_numpy.unpackb(response)
    actions = np.asarray(response['actions'])
    assert actions.shape == (10, 7) and np.isfinite(actions).all()
    with args.output.open('x') as f:
        json.dump({'task': args.task, 'mode': metadata['mode'], 'seconds': time.monotonic() - started,
                   'shape': list(actions.shape), 'finite': True,
                   'scope': 'zero-image compilation warmup; excluded from all trials and success rates'}, f, indent=2)
    print('Service warmup completed', flush=True)


if __name__ == '__main__':
    main()
