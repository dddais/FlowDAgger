"""Summarize the paired frozen-policy initial-velocity diagnostic."""
import argparse
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', required=True)
    p.add_argument('--noise_probe_runs', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--wait', action='store_true')
    args = p.parse_args()
    runs, reference = Path(args.runs).resolve(), Path(args.noise_probe_runs).resolve()
    start = time.monotonic()
    while not (runs/'queue_completed.json').exists():
        if not args.wait:
            raise RuntimeError('Queue is not terminal')
        if time.monotonic()-start > 24*3600:
            raise TimeoutError('Velocity queue still incomplete')
        time.sleep(30)
    events = [json.loads(l) for l in (runs/'events.jsonl').read_text().splitlines()]
    exits = {r['name']:r['returncode'] for r in events if r['event']=='finished'}
    rows = []
    for name, code in exits.items():
        if code != 0:
            continue
        files = list((runs/name).glob('*/noise_probe_episodes.jsonl'))
        assert len(files) == 1
        rr = [json.loads(l) for l in files[0].read_text().splitlines()]
        assert len(rr) == 50
        rows.extend(rr)
    control = {r['reset_seed']:r for r in rows if r['condition']=='default_zero_velocity'}
    treatment = {r['reset_seed']:r for r in rows if r['condition']=='training_velocity'}
    assert control.keys() == treatment.keys()
    counts = dict(both_success=0, default_only=0, velocity_only=0, both_fail=0)
    for seed, c in control.items():
        t = treatment[seed]
        assert c['initial_image_sha256'] == t['initial_image_sha256']
        assert max(abs(a-b) for a,b in zip(c['initial_qpos'],t['initial_qpos'])) < 1e-10
        dof, velocity = t['velocity_dof'], t['calibrated_velocity']
        for i,(a,b) in enumerate(zip(c['initial_qvel'],t['initial_qvel'])):
            assert abs((b-a)-(velocity if i==dof else 0.)) < 1e-10
        key = ('both_success' if c['success'] and t['success'] else 'default_only' if c['success']
               else 'velocity_only' if t['success'] else 'both_fail')
        counts[key] += 1
    ref = {}
    for file in reference.glob('metaworld_stick_push_s*/*/noise_probe_episodes.jsonl'):
        for line in file.read_text().splitlines():
            r = json.loads(line)
            if r['condition']=='gaussian':
                ref[r['reset_seed']] = r
    mismatches = []
    overlap = sorted(control.keys() & ref.keys())
    for seed in overlap:
        if any(control[seed][key] != ref[seed][key] for key in ['success','steps','initial_image_sha256']):
            mismatches.append(seed)
    n = len(control)
    result = dict(complete=len(exits)==3 and all(c==0 for c in exits.values()) and n==75,
        exits=exits,pairs=n,counts=counts,
        default_sr=(counts['both_success']+counts['default_only'])/n if n else None,
        velocity_sr=(counts['both_success']+counts['velocity_only'])/n if n else None,
        reference_noise_probe=dict(overlap=len(overlap),mismatched_reset_seeds=mismatches),episodes=rows)
    out = Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    with (out/'results.json').open('x') as f:json.dump(result,f,indent=2)
    lines=['# stick-push初始速度的Base配对诊断','','## Material Passport','',
        '- 速度来自公开训练集的状态转移，未用本轮策略评估成功率选择。冻结Base，不训练；两条件使用相同reset seed和高斯噪声序列。',
        f"- 完整方案完成：{result['complete']}；实际配对数：{n}/75。",
        f"- 原默认高斯噪声诊断的重复验证：重合{len(overlap)}对；结果/长度/图像不一致的reset seed：{mismatches}。",'']
    if result['complete']:
        lines += [f"默认零物体初速度SR：{result['default_sr']:.3f}；训练数据标定初速度SR：{result['velocity_sr']:.3f}；差值：{100*(result['velocity_sr']-result['default_sr']):+.1f}pp。",'',
            f"配对计数：双成功{counts['both_success']}；仅默认成功{counts['default_only']}；仅标定速度成功{counts['velocity_only']}；双失败{counts['both_fail']}。",'']
    lines += ['这是改变环境初始隐藏状态的诊断，不能直接替换原主表或证明论文使用相同环境。它检验公开训练数据中的物体运动与当前默认reset之间的差异是否影响Base性能。历史代码原因和论文精确设置仍需独立证据。',
        '42/43/44仅为评估分组和噪声seed，不是训练种子。失败运行保留，未筛选成功episode。']
    with (out/'RESULTS.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='episodes'}),flush=True)


if __name__=='__main__':main()
