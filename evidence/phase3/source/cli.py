"""Command-line entry point for the fixed-split research demonstration."""
import argparse
import csv
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
import yaml

from .calibration import reliability_bins
from .data import load_data
from .decision_framework import recommend
from .evaluation import binary_losses, evaluate
from .logits import probability_from_binary_score, scale_binary_score
from .models import fit_baseline
from .power_analysis import paired_required_cases
from .reporting import save_report
from .statistics import exact_mcnemar, paired_bootstrap


def run(config_path, root=Path(".")):
    root=Path(root)
    config=yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    seed=int(config["seed"])
    xtr,xte,ytr,yte=load_data(seed,float(config["test_size"]))
    model=fit_baseline(xtr,ytr,seed)
    baseline_score=model.decision_function(xte)
    challenger_score=scale_binary_score(baseline_score,float(config["challenger_logit_scale"]))
    pa=probability_from_binary_score(baseline_score)
    pb=probability_from_binary_score(challenger_score)
    la,ba=binary_losses(yte,pa)
    lb,bb=binary_losses(yte,pb)
    ci=paired_bootstrap(lb-la,replicates=int(config["bootstrap_replicates"]),
                         confidence=float(config["confidence"]),seed=seed)
    labels_a=(pa>=0.5).astype(int)
    labels_b=(pb>=0.5).astype(int)
    result={"project":"Statistical Assurance of AI Model Changes", "dataset":"sklearn breast_cancer",
            "experiment_type":"fixed trained baseline plus controlled positive logit scaling",
            "assumptions":["independent held-out subjects", "conditional case-bootstrap inference",
                            "no independent training-run variance estimate"],
            "configuration":config,"environment":{"python":platform.python_version(),
            "scikit_learn":sklearn.__version__,"numpy":np.__version__,"scipy":scipy.__version__},
            "n_train":len(ytr),"n_test":len(yte),
            "baseline_metrics":evaluate(yte,pa),"challenger_metrics":evaluate(yte,pb),
            "paired_logloss_difference_ci":ci,
            "paired_brier_difference":float((bb-ba).mean()),
            "mean_absolute_probability_change":float(np.abs(pb-pa).mean()),
            "mean_absolute_score_change":float(np.abs(challenger_score-baseline_score).mean()),
            "decision_disagreement_rate":float(np.mean(labels_a!=labels_b)),
            "mcnemar_exploratory":exact_mcnemar(yte,labels_a,labels_b),
            "reliability_baseline":reliability_bins(yte,pa),
            "reliability_challenger":reliability_bins(yte,pb)}
    result["recommendation"]=recommend(ci,len(yte),config)
    for d in ("results","reports","figures"):(root/d).mkdir(parents=True,exist_ok=True)
    (root/"results/metrics.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    with (root/"results/case_predictions.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["case_index","true_label","baseline_score","challenger_score","baseline_probability","challenger_probability","baseline_logloss","challenger_logloss"])
        for i in range(len(yte)):
            w.writerow([i,int(yte[i]),float(baseline_score[i]),float(challenger_score[i]),float(pa[i]),float(pb[i]),float(la[i]),float(lb[i])])
    fig,ax=plt.subplots(figsize=(7,6))
    for name,rows in [("Baseline",result["reliability_baseline"]),("Challenger",result["reliability_challenger"])]:
        nonempty=[r for r in rows if r["count"]]
        ax.plot([r["mean_probability"] for r in nonempty],
                [r["observed_rate"] for r in nonempty],"o-",label=name)
    ax.plot([0,1],[0,1],"k--",alpha=.5,label="Ideal")
    ax.set(xlabel="Mean predicted probability",ylabel="Observed positive fraction",title="Descriptive reliability (held-out data)",xlim=(0,1),ylim=(0,1))
    ax.legend();fig.tight_layout();fig.savefig(root/"figures/reliability.png",dpi=160);plt.close(fig)
    save_report(result,root/"reports/validation.html")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    r=sub.add_parser("run");r.add_argument("--config",default="configs/experiment.yaml")
    x=sub.add_parser("plan");x.add_argument("--sd",type=float,required=True);x.add_argument("--effect",type=float,required=True)
    x.add_argument("--alpha",type=float,default=.05);x.add_argument("--power",type=float,default=.8)
    args=parser.parse_args()
    if args.command=="run":
        out=run(args.config)
        print(json.dumps({"recommendation":out["recommendation"],"paired_logloss_difference_ci":out["paired_logloss_difference_ci"],"n_test":out["n_test"]},indent=2))
    else:
        print(json.dumps({"approximate_independent_paired_cases":paired_required_cases(args.sd,args.effect,args.alpha,args.power),"note":"Requires defensible pilot SD; not a training-run requirement"},indent=2))

if __name__=="__main__":main()
