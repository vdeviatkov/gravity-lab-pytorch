"""Post-training checkpoint replay videos; rendering stays outside the training loop."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def generate_training_videos(run_dir: Path, config: dict) -> None:
    experiment = config['experiment']
    if not experiment.get('map_overlay_after_training', True):
        return
    script = Path(__file__).resolve().parents[2] / 'scripts' / 'generate_map_overlay.py'
    command = [sys.executable, str(script), '--run-dir', str(run_dir.resolve()),
               '--seed', str(config['seeds']['final_evaluation'])]
    tracks = experiment.get('map_overlay_tracks', 'all')
    if tracks:
        command += ['--tracks', tracks]
    log_path = run_dir / 'map_overlay_generation.log'
    status_path = run_dir / 'map_overlay_status.json'
    print(f'Generating checkpoint replay videos; log: {log_path}', flush=True)
    status = {'status': 'running', 'log': str(log_path), 'command': command}
    status_path.write_text(json.dumps(status, indent=2) + '\n')
    try:
        with log_path.open('w') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as error:
        status.update(status='failed', error=str(error))
        print(f'Video generation failed; training results are saved. See {log_path}', file=sys.stderr)
    else:
        status.update(status='complete', videos=[str(p) for p in sorted(run_dir.glob('map_overlay_*.mp4'))])
        print(f'Checkpoint replay videos saved in {run_dir}', flush=True)
    finally:
        status_path.write_text(json.dumps(status, indent=2) + '\n')
