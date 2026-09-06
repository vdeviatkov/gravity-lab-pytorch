"""Regression coverage for complete terrain and camera-independent checkpoint replay."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import generate_map_overlay as overlay
from generate_map_plates import render_plate


class MapOverlayTest(unittest.TestCase):
    def test_off_map_movements_extend_canvas_without_moving_world_geometry(self):
        from PIL import Image
        plate = Image.new('RGB', (200, 100), 'white')
        plate.putpixel((0, 0), (0, 255, 0))
        paths = [[(0, -800, 700), (1, 900, -1000)]]
        expanded, left, top = overlay.fit_plate_to_paths(plate, 0, 0, paths)
        self.assertEqual(expanded.getpixel((-left, -top)), (0, 255, 0))
        for _, x, y in paths[0]:
            px, py = overlay.bike_canvas_pos(x, y, left, top)
            self.assertTrue(100 <= px <= expanded.width - 100)
            self.assertTrue(100 <= py <= expanded.height - 100)

    def test_final_policy_included_without_timelapse(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / 'final.gdp').touch()
            (run / 'summary.json').write_text('{"active_training_duration_seconds": 42}')
            self.assertEqual(overlay.checkpoint_files(run), [(42, run / 'final.gdp')])

    def test_recording_passes_simulation_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / 'positions.csv').write_text('frame,episode,bike_x,bike_y\n0,0,-320,240\n')
            with patch.object(overlay.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, 'ok')) as run:
                overlay.record_checkpoint(Path('policy.gdp'), 1, 2, 3, 90, 12, folder, 5, 'custom.mrg')
            command = run.call_args.args[0]
            for option, expected in [('--frame-skip', '5'), ('--league', '3'), ('--max-steps', '90'),
                                     ('--seed', '12'), ('--level-pack', 'custom.mrg')]:
                self.assertEqual(command[command.index(option) + 1], expected)
            self.assertIn('--bike-only', command)

    @unittest.skipUnless(overlay.VIEWER.exists(), 'classic viewer is not built')
    def test_direct_plate_covers_every_ground_segment(self):
        from gravity_lab import ClassicConfig, ClassicGravityEnv
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            path = render_plate(1, 2, Path(temp))
            metadata = json.loads(path.with_suffix('.json').read_text())
            with ClassicGravityEnv(ClassicConfig(level_group=1, track=2)) as env:
                points = env.track_polyline()
            with Image.open(path) as image:
                # Midpoints on every near-edge segment must be present, including
                # sections far beyond any recorded policy's coverage and SDL's 640px viewport.
                for (x0, y0), (x1, y1) in zip(points, points[1:]):
                    x = round((x0 + x1) / 2) - metadata['min_ox']
                    y = -round((y0 + y1) / 2) - metadata['min_oy']
                    pixels = [image.getpixel((x+dx, y+dy)) for dx in range(-2, 3) for dy in range(-2, 3)]
                    self.assertTrue(any(g > 150 and r < 20 and b < 20 for r,g,b,*_ in pixels), (x,y))

    def test_automatic_hook_reports_success_and_failure(self):
        from gravity_lab_rl.video import generate_training_videos
        cfg = {'experiment': {'timelapse_interval_seconds': 300, 'map_overlay_tracks': '1:2'},
               'seeds': {'final_evaluation': 12}}
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            with patch('gravity_lab_rl.video.subprocess.run') as command:
                generate_training_videos(run, cfg)
            self.assertEqual(json.loads((run/'map_overlay_status.json').read_text())['status'], 'complete')
            self.assertIn(str(run.resolve()), command.call_args.args[0])
            with patch('gravity_lab_rl.video.subprocess.run', side_effect=OSError('missing ffmpeg')):
                generate_training_videos(run, cfg)
            self.assertEqual(json.loads((run/'map_overlay_status.json').read_text())['status'], 'failed')

    def test_video_defaults_and_explicit_opt_out(self):
        from gravity_lab_rl.config import with_experiment_defaults
        from gravity_lab_rl.video import generate_training_videos
        source = {'experiment': {}, 'seeds': {'final_evaluation': 12}}
        cfg = with_experiment_defaults(source)
        self.assertEqual(source['experiment'], {})
        explicit = with_experiment_defaults({'experiment': {'map_overlay_tracks': 'all'}})
        self.assertEqual(explicit['experiment']['map_overlay_tracks'], 'all')
        self.assertTrue(cfg['experiment']['training_plot_after_training'])
        self.assertTrue(cfg['experiment']['map_overlay_after_training'])
        self.assertEqual(cfg['experiment']['map_overlay_tracks'], '0:0,1:0,2:0')
        self.assertEqual(cfg['experiment']['timelapse_interval_seconds'], 300)
        with tempfile.TemporaryDirectory() as temp:
            with patch('gravity_lab_rl.video.subprocess.run') as command:
                generate_training_videos(Path(temp), source)
            args = command.call_args.args[0]
            self.assertEqual(args[args.index('--tracks') + 1], '0:0,1:0,2:0')
            source['experiment']['map_overlay_after_training'] = False
            with patch('gravity_lab_rl.video.subprocess.run') as command:
                generate_training_videos(Path(temp), source)
            command.assert_not_called()
        self.assertNotIn('timelapse_interval_seconds', with_experiment_defaults(source)['experiment'])

    def test_training_plot_runs_even_when_videos_are_disabled(self):
        from gravity_lab_rl.video import generate_training_videos
        cfg = {'experiment': {'map_overlay_after_training': False},
               'seeds': {'final_evaluation': 12}}
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / 'metrics.jsonl').write_text('{}\n')
            with patch('gravity_lab_rl.video.subprocess.run') as command:
                generate_training_videos(run, cfg)
            command.assert_called_once()
            self.assertTrue(command.call_args.args[0][1].endswith('plot_progress.py'))
            status = json.loads((run / 'training_plot_status.json').read_text())
            self.assertEqual(status['status'], 'complete')
