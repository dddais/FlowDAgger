"""Opt-in dual-buffer collection study with successful pure autonomous rollouts.

Preserves the official collector, update loop, inversion, and model. Online
rollouts alternate pure autonomous/corrective; warmup episodes remain expert.
The wrapper still enforces the total rollout budget and rejected-label filter.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys

from reproduce_main_table import main as official_wrapper


class _DiscardInsertions:
    """Read-through buffer view that rejects insertions from mixed rollouts."""
    def __init__(self, buffer):
        self.buffer = buffer
        self.discarded = 0

    def __len__(self):
        return len(self.buffer)

    def __getattr__(self, name):
        return getattr(self.buffer, name)

    def insert(self, *args, **kwargs):
        self.discarded += 1

    def increment_traj_counter(self):
        pass


def filtered_pure_add(filtered_add, original_add, variant, traj,
                      online_replay_buffer, expert_buffer=None,
                      autonomous_buffer=None):
    if not variant.get('dual_buffer', 0) or autonomous_buffer is None:
        raise ValueError('Pure-autonomous collection requires --dual_buffer 1')
    if not variant.get('filter_failures', 1):
        raise ValueError('Pure-autonomous study requires successful-episode filtering')
    mixed = any(traj['intervention_flags'])
    proxy = _DiscardInsertions(autonomous_buffer) if mixed else autonomous_buffer
    counts = filtered_add(original_add, variant, traj, online_replay_buffer,
                          expert_buffer, proxy)
    counts['rejected_mixed_autonomous_chunks'] = proxy.discarded if mixed else 0
    return counts


def main():
    # Must be imported before official_wrapper captures collector/insertion functions.
    import dagger_loop
    original_collect=dagger_loop.collect_traj
    original_eval=dagger_loop.perform_control_eval
    base_eval_completed=False
    if '--repro_filter_invalid_inversions' not in sys.argv:
        raise ValueError('Use --repro_filter_invalid_inversions for this study')
    import reproduction_filters
    original_filtered_add=reproduction_filters.filtered_add
    online_count=0
    rollout_count=0

    def evaluate_once(agent, env, step, *args, **kwargs):
        nonlocal base_eval_completed
        if step == 0 and base_eval_completed:
            print('[Pure-autonomous study] identical step=0 evaluation already recorded; skipping repeat')
            return
        result=original_eval(agent, env, step, *args, **kwargs)
        if step == 0:
            base_eval_completed=True
        return result

    def collect(*args, **kwargs):
        nonlocal online_count, rollout_count
        variant=args[0]
        handler=kwargs.get('intervention_handler')
        if handler is None:
            raise ValueError('The scheduled correction study requires an intervention handler')
        warmup=bool(handler._force_immediate_takeover)
        if not warmup:
            online_count += 1
        pure=not warmup and online_count % 2 == 1
        previous_probability=handler.intervention_probability
        try:
            if pure:
                handler.intervention_probability=0.0
            traj=original_collect(*args, **kwargs)
        finally:
            handler.intervention_probability=previous_probability
        rollout_count += 1
        flags=traj['intervention_flags']
        if pure and any(flags):
            raise RuntimeError('An intended pure autonomous rollout contains intervention')
        with (Path(variant.outputdir)/'collection_protocol.jsonl').open('a') as f:
            f.write(json.dumps(dict(rollout=rollout_count,online_index=online_count,
                warmup=warmup,pure_autonomous_requested=pure,
                actually_pure=not any(flags),intervention_chunks=sum(flags),
                chunks=len(flags),success=bool(traj['is_success'])))+'\n')
        return traj

    def pure_filtered_add(original_add,variant,traj,online_replay_buffer,
                          expert_buffer=None,autonomous_buffer=None):
        counts=filtered_pure_add(original_filtered_add,original_add,variant,traj,
                                 online_replay_buffer,expert_buffer,autonomous_buffer)
        with (Path(variant.outputdir)/'pure_buffer_routing.jsonl').open('a') as f:
            f.write(json.dumps(dict(rollout=rollout_count,
                rejected_mixed_autonomous_chunks=counts['rejected_mixed_autonomous_chunks'],
                autonomous_buffer_size=len(autonomous_buffer)))+'\n')
        return counts

    dagger_loop.collect_traj=collect
    dagger_loop.perform_control_eval=evaluate_once
    reproduction_filters.filtered_add=pure_filtered_add
    official_wrapper()


if __name__=='__main__':
    main()
