"""Read-only progress for one explicitly selected evaluation tree."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--evaluation-dir', default='eval_v2')
    args = p.parse_args()
    root = args.root
    result = {}
    if (root / 'preparation.json').is_file():
        preparation = json.loads((root / 'preparation.json').read_text())
        result['checkpoint'] = preparation['checkpoint']
        receipt = Path(preparation['checkpoint']).parent / 'VERIFIED.json'
    else:
        receipt = root / 'verified_model_download/VERIFIED.json'
    # A saved verification receipt is provenance, not a fresh checksum audit.
    result['checkpoint_verification_receipt'] = str(receipt) if receipt.is_file() else None
    result['cache_shards'] = {p.name: len(list((p / 'shards').glob('*.npz'))) for p in sorted((root / 'cache').glob('task_*'))}
    result['training_steps'] = {str(p.parent.relative_to(root)): json.loads(p.read_text())['step']
                                for p in sorted(root.glob('standard_recovery_seed*/task_*/latest.json'))}
    result['evaluation_trials'] = {str(p.parent.relative_to(root)): len(p.read_text().splitlines())
                                   for p in sorted((root / args.evaluation_dir).glob('*/task_*/*/trials.jsonl'))}
    result['finished_evaluations'] = sorted(str(p.parent.relative_to(root)) for p in (root / args.evaluation_dir).glob('*/task_*/*/summary.json'))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
