"""Summarize actual evaluations without treating pending runs as successes."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import re
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # The final line may be in the middle of a concurrent append.
            continue
    return rows


def summarize(run_root):
    cfg = json.loads((ROOT / 'config/reproduction_20261028_auto/main_table.json').read_text())
    events = read_rows(run_root / 'events.jsonl')
    exits = {r['name']: r['returncode'] for r in events if r.get('event') == 'finished'}
    runs = []
    for task in cfg['tasks']:
        for seed in cfg['seeds']:
            name = f"{task['task_key']}_s{seed}"
            d = run_root / name
            evals = [r for p in d.glob('*/eval_results.jsonl') for r in read_rows(p)]
            state_path = d / 'final_training_state.json'
            final_step = json.loads(state_path.read_text())['final_step'] if state_path.exists() else 4000
            base = next((r['success_rate'] for r in evals if r['step'] == 0), None)
            final = next((r['success_rate'] for r in evals if r['step'] == final_step), None)
            rollouts = read_rows(d / 'rollouts.jsonl')
            log_path = run_root / f'{name}.log'
            log = log_path.read_text(errors='replace') if log_path.exists() else ''
            kept = [(int(a), int(b)) for a, b in re.findall(r'kept=(\d+)/(\d+)', log)]
            rejected = sum(b-a for a,b in kept)
            chunks = sum(b for a,b in kept)
            checkpoints = list(d.glob(f'*/checkpoint{final_step}'))
            complete = exits.get(name) == 0 and base is not None and final is not None and bool(checkpoints)
            runs.append(dict(task=task['task_key'], seed=seed, base=base,
                             flowdagger=final, improvement=(final-base if base is not None and final is not None else None),
                             rollouts=len(rollouts), rollout_successes=sum(r['is_success'] for r in rollouts),
                             rejected_chunks=rejected, inverted_chunks=chunks,
                             returncode=exits.get(name), complete=complete,
                             budget_matches=(len(rollouts)==50), final_step=final_step, evals=evals,
                             checkpoint=str(checkpoints[0]) if checkpoints else None))
    return cfg, runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    cfg, runs = summarize(Path(args.runs))
    (out / 'results.json').write_text(json.dumps(dict(time=time.time(), runs=runs), indent=2))
    keys = [k for k in runs[0] if k != 'evals']
    with (out / 'per_seed.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k:r[k] for k in keys} for r in runs)
    lines = ['# 主表复现结果快照', '', '## Material Passport', '',
             '- 来源：本地真实 eval_results.jsonl、rollouts.jsonl、队列退出码；不是论文数值的复制。',
             '- 状态：' + ('全部运行完成' if all(r['complete'] for r in runs) else '仍有运行未完成；不得作为最终复现结论。'),
             f"- 完成运行：{sum(r['complete'] for r in runs)}/36；实际 50 rollout：{sum(r['complete'] and r['budget_matches'] for r in runs)}/36。", '',
             '|任务|完整种子|Base 各种子|FlowDAgger 各种子|Base 均值|Flow 均值|论文 Base/Flow|Flow 差(pp)|',
             '|---|---:|---|---|---:|---:|---|---:|']
    aggregates = []
    def values(rows, key):
        return ', '.join(f"{r['seed']}:{r[key]:.2f}" if r[key] is not None else f"{r['seed']}:待完成" for r in rows)
    for task in cfg['tasks']:
        rs = [r for r in runs if r['task']==task['task_key']]
        finished = [r for r in rs if r['complete'] and r['budget_matches']]
        base = statistics.mean(r['base'] for r in finished) if len(finished)==3 else None
        final = statistics.mean(r['flowdagger'] for r in finished) if len(finished)==3 else None
        fmt = lambda x: f'{x:.3f}' if x is not None else '—'
        diff = (final-task['paper_flowdagger'])*100 if final is not None else None
        lines.append(f"|{task['task_key']}|{len(finished)}/3|{values(rs,'base')}|{values(rs,'flowdagger')}|{fmt(base)}|{fmt(final)}|{task['paper_base']:.2f}/{task['paper_flowdagger']:.2f}|{fmt(diff)}|")
        if base is not None:
            aggregates.append((base,final))
    if len(aggregates)==12:
        b=statistics.mean(x[0] for x in aggregates); f=statistics.mean(x[1] for x in aggregates)
        lines += ['',f'12 任务宏平均：Base={b:.4f}，FlowDAgger={f:.4f}，提升={(f-b)*100:.2f} pp。']
    lines += ['', '以 3 个训练种子为独立实验单位。未完成种子不填零、不外推。使用预定预算结束时的结果，不选择中途最高值；实际更新步数见 per_seed.csv 的 final_step。',
              '论文只有汇总值且没有原始种子结果，不能检验逐种子差异。完整逐种子数据见 per_seed.csv。', '',
              '注意：rejected_chunks 是官方日志标记数；原版未实际跳过这些 transition，修正版须结合 manifest 和 insertion_filter.jsonl 核实。']
    (out / 'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
