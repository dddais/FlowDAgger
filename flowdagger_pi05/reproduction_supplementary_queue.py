"""Run saved-input inversion and provisional held-out studies after the main queue."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

from reproduction_report import ROOT, read_rows, summarize
from reproduction_inversion_benchmark import METHODS


def build_jobs(main_runs, output, chunks):
    _, rows = summarize(main_runs)
    assert len(rows)==36 and all(r['complete'] and r['budget_matches'] for r in rows)
    cfg=ROOT/'config/reproduction_20261028_auto/with_heldout_provisional.json'
    tasks=[t for t in json.loads(cfg.read_text())['tasks'] if t.get('paper_table')==3]
    python=str(ROOT/'.venv-pi05/bin/python')
    jobs=[]
    # Successful smoke runs gate further launches in each family.
    methods=['perstep5',*[m for m in METHODS if m!='perstep5']]
    for method in methods:
        name='inversion_'+method
        jobs.append(dict(name=name, family='inversion', smoke=method=='perstep5',
            min_free_mib=17000 if method.startswith('adam') else 11000,
            command=[python,'-u',str(ROOT/'flowdagger_pi05/reproduction_inversion_benchmark.py'),
                'benchmark','--chunks_dir',str(chunks),'--method',method,
                '--output',str(output/name)]))
    for seed in [42,43,44]:
        hammer=next(r for r in rows if r['task']=='metaworld_hammer' and r['seed']==seed)
        for ti,task in enumerate(tasks):
            name=f"heldout_{task['task_key']}_s{seed}"
            jobs.append(dict(name=name,family='heldout',smoke=seed==42 and ti==0,
                min_free_mib=11000,
                command=[python,'-u',str(ROOT/'flowdagger_pi05/reproduction_heldout_eval.py'),
                    '--hammer_checkpoint',hammer['checkpoint'],'--repro_config',str(cfg),
                    '--repro_run_dir',str(output/name),'--task_key',task['task_key'],
                    '--seed',str(seed),'--eval_episodes','25','--save_eval_video','0']))
    return jobs


def report(output, docs, expected_jobs=None):
    docs.mkdir(parents=True,exist_ok=False)
    events=read_rows(output/'events.jsonl')
    exits={e['name']:e['returncode'] for e in events if e.get('event')=='finished'}
    inv=[]; held=[]
    for name,code in exits.items():
        if code!=0:
            continue
        if name.startswith('inversion_'):
            result=json.loads((output/name/'summary.json').read_text())
            assert result['chunks']==29
            inv.append(result)
        else:
            evaluation=list((output/name).glob('*/eval_results.jsonl'))
            assert len(evaluation)==1
            scores=read_rows(evaluation[0])
            assert len(scores)==2 and {s['step'] for s in scores}=={0,1}
            manifest=json.loads((output/name/'manifest.json').read_text())
            argv=manifest['argv']
            held.append(dict(task=argv[argv.index('--task_key')+1],
                seed=int(argv[argv.index('--seed')+1]),
                base=next(s['success_rate'] for s in scores if s['step']==0),
                flow=next(s['success_rate'] for s in scores if s['step']==1)))
    assert not inv or len({r['input_sha256'] for r in inv})==1
    expected_names = {j['name'] for j in expected_jobs} if expected_jobs is not None else None
    result=dict(inversion=inv,heldout=held,exits=exits,
                complete=(set(exits)==expected_names if expected_names is not None else len(exits)==23)
                         and all(c==0 for c in exits.values()))
    expected_inv = sum(j['family']=='inversion' for j in expected_jobs) if expected_jobs is not None else 8
    expected_held = sum(j['family']=='heldout' for j in expected_jobs) if expected_jobs is not None else 15
    with (docs/'results.json').open('x') as f:
        json.dump(result,f,indent=2)
    lines=['# pi0.5 补充实验', '', '## Material Passport', '',
        '- 数据：本地实际运行日志；论文：https://arxiv.org/html/2607.08877 。',
        '- 范围：固定自采29个assembly专家chunk的反演比较，以及Hammer训练后的5个暂定映射任务迁移评估。',
        f"- 完成：{len(inv)}/{expected_inv} 反演配置，{len(held)}/{expected_held} 迁移运行。", '',
        '## 反演', '',
        '|方法|4D MSE（完整补齐序列）|4D MSE（执行部分）|中位耗时ms|',
        '|---|---:|---:|---:|']
    for r in inv:
        lines.append(f"|{r['method']}|{r['mean_env4_mse_full_horizon']:.6f}|{r['mean_env4_mse_executed']:.6f}|{r['median_ms']:.1f}|")
    lines += ['', '误差主值不裁剪重建动作。两种时间范围均保留，不能事后选择更接近论文的指标。计时同步GPU、排除两次预热及额外度量解码，但包含官方invert内部的重建检查。共享RTX4090与论文RTX5090不可直接比较耗时。原论文29个输入未公开；本实验只能检验方法趋势。Adam默认5步梯度解码，full10为显式诊断；均用10步最终解码。', '',
        '## 暂定任务映射下的迁移', '',
        '|任务|完整种子|Base均值|Hammer→Flow均值|差pp|', '|---|---:|---:|---:|---:|']
    for task in sorted({r['task'] for r in held}):
        rs=[r for r in held if r['task']==task]
        if len(rs)==3:
            b=statistics.mean(r['base'] for r in rs); f=statistics.mean(r['flow'] for r in rs)
            lines.append(f'|{task}|3|{b:.4f}|{f:.4f}|{100*(f-b):+.2f}|')
        else:
            lines.append(f'|{task}|{len(rs)}|未完成|未完成|—|')
    lines += ['', 'Door/Drawer/Faucet/Plate/Push分别暂映射为door-open/drawer-open/faucet-open/plate-slide/push。提示词与训练集逐字对齐，但简称映射未获作者确认，因此不是表3精确复现。每个种子使用同种子的Hammer终点checkpoint；不在迁移任务继续训练或挑选checkpoint。评估沿用官方300步、25回合协议；由于提前成功会影响后续reset随机序列，Base/Flow不是逐episode配对实验。逐种子数据和失败退出码全部保留于results.json。', '',
        '这些结果不能替代主表，不能证明其他基线或真机实验已经复现。']
    with (docs/'RESULTS.md').open('x') as f:
        f.write('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--wait_queue',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--docs',required=True)
    p.add_argument('--chunks',required=True)
    p.add_argument('--gpus',default='0,7,1,3,6,5')
    p.add_argument('--families', default='inversion,heldout',
                   help='Comma-separated inversion/heldout families; supports a separately recorded repaired phase')
    p.add_argument('--wait_for', default='', help='Additional queue dependencies before GPU launch')
    p.add_argument('--inversion_methods', default='', help='Optional subset of saved-input inversion methods')
    p.add_argument('--adam_min_free_mib', type=int, default=17000)
    args=p.parse_args()
    runs=Path(args.wait_queue).resolve(); output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    gpus=[int(g) for g in args.gpus.split(',')]
    assert len(gpus)<=6 and len(gpus)==len(set(gpus))
    with (output/'events.jsonl').open('x') as events:
        def record(**row):
            row['time']=time.time(); events.write(json.dumps(row)+'\n'); events.flush()
            print(json.dumps(row),flush=True)
        start=time.monotonic()
        record(event='waiting_for_main_queue',path=str(runs))
        dependencies = [runs, *[Path(d).resolve() for d in args.wait_for.split(',') if d]]
        while not all((d/'queue_completed.json').exists() for d in dependencies):
            if time.monotonic()-start>24*3600:
                raise TimeoutError('Main queue still incomplete after 24 hours')
            time.sleep(30)
        pending=build_jobs(runs,output,Path(args.chunks).resolve())
        families = set(args.families.split(','))
        assert families and families <= {'inversion','heldout'}
        pending = [j for j in pending if j['family'] in families]
        if args.inversion_methods:
            selected = set(args.inversion_methods.split(','))
            assert selected and selected <= set(METHODS)
            pending = [j for j in pending if j['family'] != 'inversion'
                       or j['name'].removeprefix('inversion_') in selected]
            inv_jobs = [j for j in pending if j['family'] == 'inversion']
            for index, job in enumerate(inv_jobs):
                job['smoke'] = index == 0
        for job in pending:
            if job['name'].startswith('inversion_adam'):
                job['min_free_mib'] = args.adam_min_free_mib
        expected_jobs = list(pending)
        record(event='main_verified',jobs=len(pending))
        active={}; gate={}; idle_start=time.monotonic()
        while pending or active:
            query=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free',
                '--format=csv,noheader,nounits'],text=True,capture_output=True,check=True)
            free={int(row.split(',')[0]):int(row.split(',')[1]) for row in query.stdout.splitlines()}
            for gpu in gpus:
                if gpu in active:
                    continue
                candidates=[j for j in pending if (j['smoke'] or gate.get(j['family'])==0)
                            and free[gpu]>=j['min_free_mib']]
                if not candidates:
                    continue
                job=candidates[0]; pending.remove(job)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_GL='egl',
                    MUJOCO_EGL_DEVICE_ID=str(gpu),PYOPENGL_PLATFORM='egl',
                    XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='4',
                    OPENBLAS_NUM_THREADS='1',TF_NUM_INTEROP_THREADS='1',
                    TF_NUM_INTRAOP_THREADS='4',WANDB_MODE='disabled')
                log=(output/(job['name']+'.log')).open('x')
                proc=subprocess.Popen(job['command'],cwd=ROOT,env=env,stdout=log,
                    stderr=subprocess.STDOUT,start_new_session=True)
                active[gpu]=dict(job=job,proc=proc,log=log,start=time.monotonic())
                record(event='started',name=job['name'],gpu=gpu,pid=proc.pid,command=job['command'])
            for gpu,a in list(active.items()):
                code=a['proc'].poll()
                if code is None and time.monotonic()-a['start']>2*3600:
                    record(event='hard_timeout',name=a['job']['name'],gpu=gpu)
                    os.killpg(a['proc'].pid,signal.SIGTERM)
                    try:
                        code=a['proc'].wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(a['proc'].pid,signal.SIGKILL); code=a['proc'].wait()
                if code is not None:
                    job=a['job']; record(event='finished',name=job['name'],gpu=gpu,returncode=code)
                    a['log'].close(); del active[gpu]
                    if job['smoke']:
                        gate[job['family']]=code
                        if code!=0:
                            skipped=[j for j in pending if j['family']==job['family']]
                            for j in skipped:
                                record(event='skipped_failed_smoke',name=j['name']); pending.remove(j)
            if active:
                idle_start=time.monotonic()
            elif pending and time.monotonic()-idle_start>3600:
                record(event='resource_unavailable',pending=[j['name'] for j in pending])
                break
            if active or pending:
                time.sleep(10)
        report(output,Path(args.docs).resolve(),expected_jobs)
        record(event='supplementary_queue_ended')
        with (output/'queue_completed.json').open('x') as f:
            json.dump(dict(time=time.time(),note='Inspect results.complete; terminal queue does not imply all studies succeeded.'),f)


if __name__=='__main__':
    main()
