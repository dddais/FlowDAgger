"""Evaluate Base and a Hammer steering checkpoint on a held-out task."""
import argparse
import json
from pathlib import Path
import re
import sys

from reproduce_main_table import main as official_wrapper


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--hammer_checkpoint',required=True)
    args, remaining=p.parse_known_args()
    checkpoint=Path(args.hammer_checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    import dagger_loop
    def evaluate(variant, steering_policy, env, eval_env, replay_buffer,
                 wandb_logger, shard_fn=None, agent_dp=None,
                 intervention_handler=None, init_states=None, raw_model=None,
                 autonomous_buffer=None):
        from train_utils import perform_control_eval
        perform_control_eval(steering_policy,eval_env,0,variant,wandb_logger,
                             agent_dp,init_states=init_states,raw_model=raw_model)
        from flax.training import checkpoints
        match = re.fullmatch(r'checkpoint(\d+)', checkpoint.name)
        if match is None:
            raise ValueError(f'Expected an exact checkpoint<step> path: {checkpoint}')
        latest = checkpoints.latest_checkpoint(str(checkpoint.parent), prefix='checkpoint')
        if latest is None or Path(latest).resolve() != checkpoint:
            raise RuntimeError(f'Requested checkpoint is not the latest in its parent: {checkpoint}; latest={latest}')
        steering_policy.restore_checkpoint(str(checkpoint.parent))
        if int(steering_policy._actor.step) != int(match.group(1)):
            raise RuntimeError('Restored actor step does not match the requested checkpoint')
        perform_control_eval(steering_policy,eval_env,1,variant,wandb_logger,
                             agent_dp,init_states=init_states,raw_model=raw_model)
        with (Path(variant.outputdir)/'transfer_source.json').open('x') as f:
            json.dump(dict(checkpoint=str(checkpoint),source_task='hammer-v3',
                target_task=variant.task_key,training_rollouts=0,
                mapping_status='provisional paper shorthand mapping'),f,indent=2)
    dagger_loop.flowdagger_training_loop=evaluate
    sys.argv=[sys.argv[0],*remaining]
    official_wrapper()


if __name__=='__main__':
    main()
