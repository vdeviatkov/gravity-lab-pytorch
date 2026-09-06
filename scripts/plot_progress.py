#!/usr/bin/env python3
"""Plot evaluation quality, training dynamics, and map exposure for a saved run."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'artifacts' / '.matplotlib'))
from gravity_lab_rl.control import resolve_run


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def rolling(rows, field, window):
    times, values = [], []
    for i, row in enumerate(rows):
        recent = rows[max(0, i-window+1):i+1]
        times.append(row['active_training_seconds']/60)
        values.append(sum(r.get(field, r.get('progress', 0)) for r in recent)/len(recent))
    return times, values


def finished_maps(evaluation):
    return len({(r['level_group'], r['track']) for r in evaluation['episodes'] if r['finished']})


def evaluation_points(run, summary):
    points = [(r['active_training_seconds']/60, r['evaluation'])
              for r in read_jsonl(run/'evaluation_history.jsonl')]
    if summary.get('final_evaluation'):
        final = (summary['active_training_duration_seconds']/60, summary['final_evaluation'])
        if not points or points[-1][0] != final[0]:
            points.append(final)
    return points


def plot_run(run, output=None, window=10, best_score=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter
    import numpy as np
    rows = read_jsonl(run/'metrics.jsonl')
    if not rows:
        raise ValueError(f'No completed training episodes in {run}')
    summary_path=run/'summary.json'
    summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
    points=evaluation_points(run, summary)
    baseline_path=run/'baseline_600s_evaluation.json'
    baseline=json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    full=[r for r in rows if not r.get('practice_prefix_steps',0)]
    practice=[r for r in rows if r.get('practice_prefix_steps',0)]
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                         'axes.titleweight':'bold','font.family':'DejaVu Sans'})
    fig, axes=plt.subplots(3,2,figsize=(15,12),layout='constrained')
    blue,orange,green='#2463A5','#D97924','#278462'
    duration=summary.get('active_training_duration_seconds',rows[-1]['active_training_seconds'])/60
    fig.suptitle(f'Training results — {duration:.1f} minutes\n{run.name}',fontsize=16)
    for ax in axes.flat:
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
    ax=axes[0,0]
    if points:
        ax.plot([t for t,e in points],[finished_maps(e) for t,e in points],'-o',color=blue,label='Fixed full-start evaluation')
    if baseline:
        ax.axhline(finished_maps(baseline),ls='--',color='#777777',label='Old setup at 10 min')
    if best_score is not None:
        ax.axhline(best_score,ls=':',color=green,label='Best reference')
    ax.set(title='Maps finished in evaluation',ylabel='Maps finished (out of 30)',xlabel='Training time (min)',xlim=(0,max(1,duration)))
    ax.set_ylim(0,max(2,max([finished_maps(e) for t,e in points]+[finished_maps(baseline) if baseline else 0,best_score or 0])+1))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True));ax.legend(fontsize=8)
    ax=axes[0,1]
    if points:ax.plot([t for t,e in points],[e['mean_progress'] for t,e in points],'-o',color=blue,label='Fixed full-start evaluation')
    if baseline:ax.axhline(baseline['mean_progress'],ls='--',color='#777777',label='Old setup at 10 min')
    ax.set(title='Mean progress at episode end',ylabel='Mean map progress',xlabel='Training time (min)',xlim=(0,max(1,duration)),ylim=(-.1,1))
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(fontsize=8)
    for ax,field,title in [(axes[1,0],'reward','Training reward'),(axes[1,1],'peak_progress','Peak progress during training')]:
        for subset,label,color in [(full,'Full-start episodes',blue),(practice,'Practice suffixes',orange)]:
            if subset:ax.plot(*rolling(subset,field,window),color=color,label=label)
        ax.set(title=f'{title} — rolling {window} episodes',xlabel='Training time (min)',xlim=(0,max(1,duration)))
        ax.set_ylabel('Episode reward' if field=='reward' else 'Peak map progress')
        if field!='reward':
            ax.set_ylim(-.1,1.05);ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(fontsize=8)
    ax=axes[2,0]
    for g,color in enumerate([blue,orange,green]):
        seen=set();times=[0];counts=[0]
        for row in full:
            if row['level_group']==g:seen.add(row['track'])
            times.append(row['active_training_seconds']/60);counts.append(len(seen))
        ax.step(times,counts,where='post',color=color,label=f'Group {g}')
    ax.set(title='Maps with completed full-start training episodes',xlabel='Training time (min)',ylabel='Distinct maps per group',ylim=(0,10.5),xlim=(0,max(1,duration)))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True));ax.legend(fontsize=8)
    ax=axes[2,1];labels=[f'{g}:{t}' for g in range(3) for t in range(10)]
    full_counts=[sum((r['level_group'],r['track'])==(g,t) for r in full) for g in range(3) for t in range(10)]
    practice_counts=[sum((r['level_group'],r['track'])==(g,t) for r in practice) for g in range(3) for t in range(10)]
    x=np.arange(30)
    ax.bar(x,full_counts,color=blue,label='Full start')
    ax.bar(x,practice_counts,bottom=full_counts,color=orange,label='Practice suffix')
    ax.set_xticks(x,labels,rotation=90,fontsize=7)
    ax.set(title='Completed training episodes per map',ylabel='Episodes',xlabel='Group : track')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True));ax.legend(fontsize=8)
    output=output or run/'progress.png';output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=160)
    fig.savefig(output.with_suffix('.svg'))
    plt.close(fig)
    print(f'Wrote {output}',flush=True)
    return output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group()
    source.add_argument('--run-id');source.add_argument('--run-dir',type=Path)
    parser.add_argument('--latest',action='store_true')
    parser.add_argument('--window',type=int,default=10)
    parser.add_argument('--best-score',type=int)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    if args.window<1:parser.error('--window must be positive')
    run=args.run_dir.resolve() if args.run_dir else resolve_run(args.run_id,args.latest or args.run_id is None)
    plot_run(run,args.output,args.window,args.best_score)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
