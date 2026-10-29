"""Validate and summarize the fixed-reset Base noise diagnostic."""
import argparse
import json
from pathlib import Path
import time

from reproduction_base_noise_probe import reset_seeds


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--wait', action='store_true')
    args = p.parse_args()
    runs = Path(args.runs).resolve()
    start = time.monotonic()
    while not (runs / 'queue_completed.json').exists():
        if not args.wait:
            raise RuntimeError('Queue is still running')
        if time.monotonic() - start > 24 * 3600:
            raise TimeoutError('Diagnostic queue still running')
        time.sleep(30)
    events = [json.loads(x) for x in (runs / 'events.jsonl').read_text().splitlines()]
    exits = {e['name']: e['returncode'] for e in events if e['event'] == 'finished'}
    rows = []
    for name, code in exits.items():
        if code != 0:
            continue
        files = list((runs / name).glob('*/noise_probe_episodes.jsonl'))
        assert len(files) == 1
        records = [json.loads(x) for x in files[0].read_text().splitlines()]
        assert len(records) == 50
        seed = records[0]['seed']
        expected = reset_seeds(seed, 25)
        for reset_seed in expected:
            pair = [r for r in records if r['reset_seed'] == reset_seed]
            assert len(pair) == 2 and {r['condition'] for r in pair} == {'gaussian', 'zero'}
            assert pair[0]['initial_image_sha256'] == pair[1]['initial_image_sha256']
            assert len(pair[0]['initial_raw']) == len(pair[1]['initial_raw'])
            assert all(abs(a-b) <= 1e-6 for a,b in zip(pair[0]['initial_raw'], pair[1]['initial_raw']))
        rows.extend(records)
    tasks = []
    for task in ('metaworld_stick_push', 'metaworld_assembly'):
        rs = [r for r in rows if r['task'] == task]
        counts = dict(both_success=0, gaussian_only=0, zero_only=0, both_fail=0)
        for reset_seed in sorted({r['reset_seed'] for r in rs}):
            pair = {r['condition']: r for r in rs if r['reset_seed'] == reset_seed}
            assert len(pair) == 2
            g, z = pair['gaussian']['success'], pair['zero']['success']
            category = ('both_success' if g and z else 'gaussian_only' if g
                        else 'zero_only' if z else 'both_fail')
            counts[category] += 1
        n = sum(counts.values())
        tasks.append(dict(task=task, pairs=n, counts=counts,
            gaussian=(counts['both_success'] + counts['gaussian_only']) / n if n else None,
            zero=(counts['both_success'] + counts['zero_only']) / n if n else None,
            difference=(counts['zero_only'] - counts['gaussian_only']) / n if n else None))
    complete = len(exits) == 6 and all(c == 0 for c in exits.values()) and all(t['pairs'] == 75 for t in tasks)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'results.json').open('x') as f:
        json.dump(dict(complete=complete, exits=exits, tasks=tasks, episodes=rows), f, indent=2)
    lines = ['# 固定初始状态的Base噪声诊断结果', '', '## Material Passport', '',
        f'- 来源：本地实际逐episode结果。完整方案完成：{complete}。',
        '- 方案见 ../BASE_NOISE_DIAGNOSTIC_PROTOCOL.md；无训练，全部评估冻结Base。', '',
        '|任务|配对数|高斯SR|零噪声SR|零−高斯pp|双成功|仅高斯|仅零|双失败|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for t in tasks:
        if t['pairs'] != 75:
            lines.append(f"|{t['task']}|{t['pairs']}|未完成|未完成|—|—|—|—|—|")
            continue
        c = t['counts']
        lines.append(f"|{t['task']}|75|{t['gaussian']:.3f}|{t['zero']:.3f}|{100*t['difference']:+.1f}|{c['both_success']}|{c['gaussian_only']}|{c['zero_only']}|{c['both_fail']}|")
    lines += ['', '75个reset seed均预先固定，两个任务使用相同seed编号，但各自环境的初始状态不同。42/43/44是工作进程及噪声分组标签，不是训练种子。',
        '这套独立reset协议与主表官方评估不同，不能把较高分替代主表Base。两条件共用初始状态与图像的核验已通过；结果描述本批状态对latent采样方式的敏感性，不证明论文使用零噪声或已解释全部差距。',
        '本报告不对失败运行做隐式重试；退出码及所有已完成episode保留于results.json。']
    with (out / 'RESULTS.md').open('x') as f:
        f.write('\n'.join(lines) + '\n')
    print(json.dumps(dict(complete=complete, tasks=tasks)), flush=True)


if __name__ == '__main__':
    main()
