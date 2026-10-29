"""Opt-in correction of rejected-inversion buffer insertion.

No trajectory compaction: next observations retain their original adjacency.
The official perstep_fp collector stores accepted expert indices in
full_noise_list; intervention indices absent from that mapping are rejected.
"""
from __future__ import annotations


class _InsertionFilter:
    def __init__(self, buffer, rejected, counter):
        self._buffer = buffer
        self._rejected = rejected
        self._counter = counter

    def __getattr__(self, name):
        return getattr(self._buffer, name)

    def __len__(self):
        return len(self._buffer)

    def insert(self, transition, **kwargs):
        index = self._counter['index']
        self._counter['index'] += 1
        if index in self._rejected:
            self._counter['skipped'] += 1
            return
        return self._buffer.insert(transition, **kwargs)


def filtered_add(original_add, variant, traj, online_replay_buffer,
                 expert_buffer=None, autonomous_buffer=None):
    if variant.get('inversion_method') != 'perstep_fp':
        raise ValueError('Rejected-inversion filtering requires perstep_fp.')
    if variant.get('filter_autonomous', 0):
        raise ValueError('This insertion-index filter requires filter_autonomous=0.')
    flags = traj.get('intervention_flags', [])
    if len(flags) != len(traj['actions']):
        raise ValueError('Intervention flags and action chunks must align.')
    accepted = traj.get('full_noise_list', {})
    rejected = {i for i, flag in enumerate(flags) if flag and i not in accepted}
    counter = {'index': 0, 'skipped': 0}
    online_proxy = _InsertionFilter(online_replay_buffer, rejected, counter)
    auto_proxy = (_InsertionFilter(autonomous_buffer, rejected, counter)
                  if autonomous_buffer is not None else None)
    original_add(variant, traj, online_proxy, expert_buffer=expert_buffer,
                 autonomous_buffer=auto_proxy)
    assert counter['index'] == len(traj['actions'])
    assert counter['skipped'] == len(rejected)
    return counter
