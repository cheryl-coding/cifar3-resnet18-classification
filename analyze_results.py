"""Reproduce paired held-out accuracy comparison and error analyses from saved CSVs.

Usage: python analyze_results.py --results runs
Requires NumPy and Matplotlib. No model checkpoint or CIFAR download required.
"""
import argparse
import csv
import json
from math import comb
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASSES = ['cat', 'dog', 'horse']

def load_predictions(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

def calculate(results: Path, samples=2000, seed=42):
    frozen = load_predictions(results / 'frozen' / 'test_predictions.csv')
    tuned = load_predictions(results / 'fine_tune' / 'test_predictions.csv')
    if len(frozen) != len(tuned) or not frozen:
        raise ValueError('Expected two nonempty, equal-sized prediction files')
    if any(a['true'] != b['true'] or a['test_position'] != b['test_position'] for a,b in zip(frozen,tuned)):
        raise ValueError('Predictions are not aligned to the same ordered test images')
    correct_a = np.array([r['true'] == r['pred'] for r in frozen], dtype=bool)
    correct_b = np.array([r['true'] == r['pred'] for r in tuned], dtype=bool)
    d = correct_b.astype(int) - correct_a.astype(int)
    rng = np.random.default_rng(seed)
    boot = np.array([d[rng.integers(len(d),size=len(d))].mean() for _ in range(samples)])
    gained = int(np.sum(~correct_a & correct_b))
    lost = int(np.sum(correct_a & ~correct_b))
    n = gained + lost
    # Exact two-sided McNemar p-value, equivalent to a sign/binomial test on discordant pairs.
    p = min(1., 2 * sum(comb(n,k) for k in range(min(gained,lost)+1)) / (2**n)) if n else 1.
    per_class = {}
    for c in CLASSES:
        idx=np.array([i for i,x in enumerate(frozen) if x['true']==c])
        per_class[c]={'support':len(idx),'frozen_correct':int(correct_a[idx].sum()),
                      'fine_tune_correct':int(correct_b[idx].sum()),
                      'errors_fixed':int(np.sum(~correct_a[idx]&correct_b[idx])),
                      'new_errors':int(np.sum(correct_a[idx]&~correct_b[idx]))}
    output={'test_count':len(d),'frozen_correct':int(correct_a.sum()),'fine_tune_correct':int(correct_b.sum()),
            'both_correct':int(np.sum(correct_a&correct_b)),'both_wrong':int(np.sum(~correct_a&~correct_b)),
            'frozen_wrong_fine_tune_correct':gained,'frozen_correct_fine_tune_wrong':lost,
            'accuracy_difference':float(d.mean()),
            'accuracy_difference_95_ci_paired_bootstrap':[float(q) for q in np.quantile(boot,[.025,.975])],
            'paired_bootstrap_resamples':samples,'paired_bootstrap_seed':seed,
            'exact_mcnemar_two_sided_p':float(p),'class_changes':per_class}
    return output

def plot_results(results, summary):
    a=json.loads((results/'frozen'/'test_metrics.json').read_text())
    b=json.loads((results/'fine_tune'/'test_metrics.json').read_text())
    plt.rcParams.update({'font.size':10})
    fig,ax=plt.subplots(figsize=(8,4.4));x=np.arange(2);w=.32
    ax.bar(x-w/2,[100*a['accuracy'],100*a['macro_f1']],width=w,label='Frozen head',color='#7498b5')
    ax.bar(x+w/2,[100*b['accuracy'],100*b['macro_f1']],width=w,label='Fine-tune layer4',color='#174d76')
    for i in range(2):
        ax.text(x[i]-w/2,[100*a['accuracy'],100*a['macro_f1']][i]+1,f"{[100*a['accuracy'],100*a['macro_f1']][i]:.1f}%",ha='center',fontsize=10)
        ax.text(x[i]+w/2,[100*b['accuracy'],100*b['macro_f1']][i]+1,f"{[100*b['accuracy'],100*b['macro_f1']][i]:.1f}%",ha='center',fontsize=10)
    ax.set_xticks(x,['Accuracy','Macro-F1']);ax.set_ylim(0,108);ax.set_ylabel('Test-set score (%)')
    ax.spines[['top','right']].set_visible(False);ax.legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,1.14))
    fig.tight_layout();fig.savefig(results/'model_comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.4));x=np.arange(3)
    af=[100*a['class_report'][k]['recall'] for k in CLASSES]; bf=[100*b['class_report'][k]['recall'] for k in CLASSES]
    ax.bar(x-.18,af,width=.36,label='Frozen head',color='#7498b5');ax.bar(x+.18,bf,width=.36,label='Fine-tune layer4',color='#174d76')
    ax.set_xticks(x,CLASSES);ax.set_ylim(0,110);ax.set_ylabel('Recall (%)');ax.spines[['top','right']].set_visible(False)
    for i in range(3):
        ax.text(x[i]-.18,af[i]+1.4,f'{af[i]:.1f}',ha='center',fontsize=9)
        ax.text(x[i]+.18,bf[i]+1.4,f'{bf[i]:.1f}',ha='center',fontsize=9)
    ax.legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,1.14));fig.tight_layout()
    fig.savefig(results/'class_recall_comparison.png',dpi=180);plt.close(fig)
    return a,b

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--results',default='runs');a=parser.parse_args()
    root=Path(a.results); summary=calculate(root)
    (root/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    plot_results(root,summary)
    print(json.dumps(summary,indent=2))
