#!/usr/bin/env python3
"""Generate a run's results plot and map videos without starting training."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gravity_lab_rl.config import with_experiment_defaults
from gravity_lab_rl.control import resolve_run
from gravity_lab_rl.video import generate_training_videos


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run-id')
    source.add_argument('--run-dir', type=Path)
    parser.add_argument('--tracks', help='group:track pairs or all; default uses run settings')
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--source', choices=['auto', 'training', 'checkpoints'], default='auto')
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    run = args.run_dir.resolve() if args.run_dir else resolve_run(args.run_id, False)
    config = with_experiment_defaults(json.loads((run / 'config.json').read_text()))
    experiment = config['experiment']
    experiment.update(training_plot_after_training=True, map_overlay_after_training=True,
                      map_overlay_batch_size=args.batch_size, map_overlay_source=args.source)
    if args.tracks is not None:
        experiment['map_overlay_tracks'] = args.tracks
    generate_training_videos(run, config)
    statuses = [run / 'map_overlay_status.json', run / 'training_plot_status.json']
    return int(any(p.exists() and json.loads(p.read_text())['status'] == 'failed' for p in statuses))


if __name__ == '__main__':
    raise SystemExit(main())
