"""OpenRCA ContextControl-v1 replay experiment.

This is a source-acquisition experiment, not a new anomaly detector.  It turns
the locked 51-case OpenRCA Telecom artifact into paired replay episodes where
the hidden incident is fixed and a policy chooses which source summaries to
acquire before returning a component shortlist or abstaining.

The v1 action is component triage (top-3 shortlist).  Reason and occurrence
time are deliberately excluded because the locked baseline has no competent
action model for those criteria.
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_ORDER = ("metric", "trace", "app", "middleware", "metric_duplicate")
SOURCE_COST = {"metric": 1.0, "trace": 2.0, "app": 1.5, "middleware": 1.5, "metric_duplicate": 0.8}


@dataclass(frozen=True)
class Source:
    name: str
    candidates: tuple[str, ...]
    cost: float
    latency: float
    age_minutes: float = 0.0
    entity_match: int = 1
    available: int = 1
    provenance_group: str = ""


@dataclass
class Episode:
    case_id: int
    date: str
    level: str
    truth: str
    prior: tuple[str, ...]
    combined: tuple[str, ...]
    sources: dict[str, Source]
    condition: str


def parse_list(value) -> tuple[str, ...]:
    if pd.isna(value) or value == "":
        return ()
    parsed = json.loads(value) if isinstance(value, str) and value.startswith("[") else [value]
    return tuple(str(item) for item in parsed if str(item))


def date_split(frame: pd.DataFrame) -> dict[str, set[str]]:
    dates = sorted(frame.date.unique())
    # Chronological groups prevent telemetry from the same date appearing in
    # more than one partition.
    return {"train": set(dates[:6]), "validation": set(dates[6:8]), "test": set(dates[8:])}


def level_priors(train: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    priors = {}
    for level, group in train.groupby("level"):
        priors[level] = tuple(group.actual_component.value_counts().index[:3])
    return priors


def base_episode(row, prior) -> Episode:
    metric = parse_list(row.metric_top3)
    trace = (str(row.trace_top1),) if pd.notna(row.trace_top1) and str(row.trace_top1) else ()
    app_signal = ()
    middleware_signal = ()
    sources = {
        "metric": Source("metric", metric, 1.0, 1.0, provenance_group="metric-current"),
        "trace": Source("trace", trace, 2.0, 3.0, provenance_group="trace-current"),
        "app": Source("app", app_signal, 1.5, 2.0, provenance_group="app-current"),
        "middleware": Source("middleware", middleware_signal, 1.5, 2.0, provenance_group="middleware-current"),
        "metric_duplicate": Source("metric_duplicate", metric, 0.8, 1.0, provenance_group="metric-current"),
    }
    return Episode(int(row.record_index), str(row.date), str(row.level), str(row.actual_component), prior, parse_list(row.combined_top3), sources, "canonical")


def donor_for(episodes, episode, offset=1):
    pool = [item for item in episodes if item.level == episode.level and item.case_id != episode.case_id]
    return pool[(episode.case_id + offset) % len(pool)] if pool else episode


def conditions(base: list[Episode]) -> list[Episode]:
    output = []
    for episode in base:
        output.append(episode)
        trace_unavailable = dict(episode.sources)
        trace_unavailable["trace"] = replace(trace_unavailable["trace"], available=0)
        output.append(replace(episode, sources=trace_unavailable, condition="trace_unavailable"))

        stale = dict(episode.sources)
        donor = donor_for(base, episode)
        stale["metric"] = replace(stale["metric"], candidates=donor.sources["metric"].candidates, age_minutes=1440.0, provenance_group="metric-stale")
        output.append(replace(episode, sources=stale, condition="stale_metric"))

        wrong = dict(episode.sources)
        wrong["metric"] = replace(wrong["metric"], candidates=donor.sources["metric"].candidates, entity_match=0, provenance_group="metric-other-entity")
        output.append(replace(episode, sources=wrong, condition="wrong_entity_metric"))

        expensive = dict(episode.sources)
        expensive["trace"] = replace(expensive["trace"], cost=8.0, latency=12.0)
        output.append(replace(episode, sources=expensive, condition="high_trace_cost"))

        gap = dict(episode.sources)
        gap["metric"] = replace(gap["metric"], age_minutes=60.0)
        gap["trace"] = replace(gap["trace"], age_minutes=60.0)
        output.append(replace(episode, sources=gap, condition="temporal_gap"))
    return output


def action(episode: Episode, acquired: set[str]) -> tuple[str, ...]:
    usable = [episode.sources[name] for name in SOURCE_ORDER if name in acquired and episode.sources[name].available]
    current_metric = "metric" in acquired and episode.sources["metric"].available and episode.sources["metric"].age_minutes == 0 and episode.sources["metric"].entity_match
    current_trace = "trace" in acquired and episode.sources["trace"].available and episode.sources["trace"].age_minutes == 0 and episode.sources["trace"].entity_match
    if current_metric and current_trace:
        return episode.combined[:3]
    candidates = []
    for source in usable:
        for candidate in source.candidates:
            if candidate not in candidates:
                candidates.append(candidate)
    for candidate in episode.prior:
        if candidate not in candidates:
            candidates.append(candidate)
    return tuple(candidates[:3])


def observable_features(episode: Episode, acquired: set[str]) -> np.ndarray:
    sources = [episode.sources[name] for name in SOURCE_ORDER if name in acquired and episode.sources[name].available]
    candidates = [candidate for source in sources for candidate in source.candidates]
    unique = set(candidates)
    provenance = {source.provenance_group for source in sources}
    total_cost = sum(source.cost for source in sources)
    total_latency = sum(source.latency for source in sources)
    age = max([source.age_minutes for source in sources] or [0.0])
    entity_match = min([source.entity_match for source in sources] or [1])
    values = [
        1.0,
        float(episode.level == "node"), float(episode.level == "pod"), float(episode.level == "service"),
        *[float(name in acquired and episode.sources[name].available) for name in SOURCE_ORDER],
        len(sources), len(candidates), len(unique), len(provenance),
        float(len(candidates) > len(unique)), float(len(unique) == 1 and len(candidates) > 1),
        total_cost, total_latency, age, float(entity_match),
        float(any(source.age_minutes > 0 for source in sources)),
        float(any(not source.entity_match for source in sources)),
        float(episode.sources["trace"].available),
    ]
    return np.asarray(values, dtype=float)


class Logistic:
    def __init__(self, l2=1.0, steps=3000, rate=0.05):
        self.l2, self.steps, self.rate = l2, steps, rate

    def fit(self, x, y):
        self.mean = x[:, 1:].mean(0); self.scale = x[:, 1:].std(0); self.scale[self.scale == 0] = 1
        z = x.copy(); z[:, 1:] = (z[:, 1:] - self.mean) / self.scale
        self.w = np.zeros(z.shape[1]); y = np.asarray(y, dtype=float)
        pos = max(y.sum(), 1); neg = max(len(y) - y.sum(), 1)
        weights = np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg))
        for _ in range(self.steps):
            p = 1 / (1 + np.exp(-np.clip(z @ self.w, -30, 30)))
            grad = z.T @ ((p - y) * weights) / len(y)
            grad[1:] += self.l2 * self.w[1:] / len(y)
            self.w -= self.rate * grad
        return self

    def predict(self, x):
        z = np.asarray(x, dtype=float).copy()
        if z.ndim == 1: z = z[None, :]
        z[:, 1:] = (z[:, 1:] - self.mean) / self.scale
        return 1 / (1 + np.exp(-np.clip(z @ self.w, -30, 30)))


def states(episodes):
    rows = []
    for episode in episodes:
        available = [name for name in SOURCE_ORDER if episode.sources[name].available]
        for size in range(len(available) + 1):
            for subset in itertools.combinations(available, size):
                acquired = set(subset); shortlist = action(episode, acquired)
                rows.append((episode, acquired, observable_features(episode, acquired), int(episode.truth in shortlist)))
    return rows


def auc(y, p):
    y=np.asarray(y); p=np.asarray(p); pos=p[y==1]; neg=p[y==0]
    if not len(pos) or not len(neg): return float("nan")
    return float(np.mean([(a>b)+.5*(a==b) for a in pos for b in neg]))


def average_precision(y, p):
    order=np.argsort(-np.asarray(p)); y=np.asarray(y)[order]; total=y.sum()
    return float(sum(y[i] * y[:i+1].mean() for i in range(len(y))) / total) if total else 0.0


def ece(y, p, bins=10):
    y=np.asarray(y); p=np.asarray(p); result=0.0
    for lo in np.linspace(0,1,bins,endpoint=False):
        mask=(p>=lo)&(p<lo+1/bins if lo+1/bins<1 else p<=1)
        if mask.any(): result += mask.mean()*abs(p[mask].mean()-y[mask].mean())
    return float(result)


def select_threshold(model, validation_states):
    y=np.array([row[3] for row in validation_states]); p=model.predict(np.vstack([row[2] for row in validation_states]))
    best=(float("-inf"), .5)
    for threshold in np.linspace(.1,.9,17):
        act=p>=threshold
        utility=((y*act) - 1.5*((1-y)*act) + .1*(~act)).mean()
        if utility>best[0]: best=(utility,float(threshold))
    return best[1]


def next_source(model, episode, acquired, cost_weight=.03):
    current=float(model.predict(observable_features(episode,acquired))[0]); choices=[]
    for name in SOURCE_ORDER:
        source=episode.sources[name]
        if name in acquired or not source.available: continue
        post=float(model.predict(observable_features(episode,acquired|{name}))[0])
        choices.append((post-current-cost_weight*source.cost, post, -source.cost, name))
    return max(choices)[-1] if choices else None


def ranker_features(episode: Episode, acquired: set[str], source_name: str) -> np.ndarray:
    """Features available before acquisition: state plus source-registry metadata."""
    source = episode.sources[source_name]
    registry = [
        *[float(source_name == name) for name in SOURCE_ORDER],
        source.cost, source.latency, source.age_minutes, float(source.entity_match),
        float(source.available), float(source.provenance_group in {episode.sources[n].provenance_group for n in acquired}),
    ]
    return np.concatenate([observable_features(episode, acquired), np.asarray(registry, dtype=float)])


def ranker_training(episodes):
    x, y = [], []
    for episode, acquired, _, _ in states(episodes):
        for source_name in SOURCE_ORDER:
            if source_name in acquired or not episode.sources[source_name].available:
                continue
            x.append(ranker_features(episode, acquired, source_name))
            y.append(int(episode.truth in action(episode, acquired | {source_name})))
    return np.vstack(x), np.asarray(y)


def ranked_source(ranker, episode, acquired, cost_weight=.03):
    choices=[]
    for source_name in SOURCE_ORDER:
        source=episode.sources[source_name]
        if source_name in acquired or not source.available: continue
        probability=float(ranker.predict(ranker_features(episode,acquired,source_name))[0])
        choices.append((probability-cost_weight*source.cost,probability,-source.cost,source_name))
    return max(choices)[-1] if choices else None


def run_policy(name, episodes, model, ranker, threshold, rng):
    results=[]
    for episode in episodes:
        available=[s for s in SOURCE_ORDER if episode.sources[s].available]
        acquired=set(); abstain=False
        if name=="always_fetch": acquired=set(available)
        elif name=="fixed_top1": acquired={"metric"} if episode.sources["metric"].available else set()
        elif name=="random_top1": acquired={rng.choice(available)} if available else set()
        elif name=="freshness_only":
            fresh=[s for s in available if episode.sources[s].age_minutes==0]
            acquired={min(fresh,key=lambda s:episode.sources[s].cost)} if fresh else set()
        elif name in {"source_ranker", "gate_plus_ranker"}:
            max_reads=2 if name=="source_ranker" else len(available)
            while len(acquired)<max_reads:
                probability=float(model.predict(observable_features(episode,acquired))[0])
                if name=="gate_plus_ranker" and probability>=threshold: break
                source=ranked_source(ranker,episode,acquired)
                if source is None: break
                acquired.add(source)
            abstain=name=="gate_plus_ranker" and float(model.predict(observable_features(episode,acquired))[0])<threshold
        elif name=="oracle":
            options=[]
            for size in range(len(available)+1):
                for subset in itertools.combinations(available,size):
                    subset=set(subset)
                    if episode.truth in action(episode,subset):
                        options.append((sum(episode.sources[s].cost for s in subset),len(subset),subset))
            acquired=min(options,key=lambda x:(x[0],x[1]))[2] if options else set(available); abstain=not options
        shortlist=action(episode,acquired); correct=int(episode.truth in shortlist and not abstain)
        results.append({"policy":name,"case_id":episode.case_id,"condition":episode.condition,"correct":correct,"abstain":int(abstain),"calls":len(acquired),"cost":sum(episode.sources[s].cost for s in acquired),"latency":sum(episode.sources[s].latency for s in acquired),"selected":"|".join(sorted(acquired)),"shortlist":"|".join(shortlist)})
    return results


def main(data: Path, output: Path, seed: int):
    frame=pd.read_csv(data); dates=sorted(frame.date.unique()); rng=np.random.default_rng(seed)
    policies=["no_context","always_fetch","fixed_top1","random_top1","freshness_only","source_ranker","gate_plus_ranker","oracle"]
    rows=[]; state_y=[]; state_p=[]; thresholds=[]; fold_manifest=[]
    # Leave-one-date-out evaluation. Two preceding dates calibrate the stopping
    # threshold; all remaining dates train both decision-level objectives.
    for test_index,test_date in enumerate(dates):
        validation_dates={dates[(test_index-1)%len(dates)],dates[(test_index-2)%len(dates)]}
        train_dates=set(dates)-validation_dates-{test_date}
        priors=level_priors(frame[frame.date.isin(train_dates)])
        base=[base_episode(row,priors.get(row.level,())) for row in frame.itertuples()]
        all_episodes=conditions(base)
        train_episodes=[e for e in all_episodes if e.date in train_dates]
        validation_episodes=[e for e in all_episodes if e.date in validation_dates]
        test_episodes=[e for e in all_episodes if e.date==test_date]
        train_states=states(train_episodes); validation_states=states(validation_episodes); test_states=states(test_episodes)
        model=Logistic().fit(np.vstack([r[2] for r in train_states]),np.array([r[3] for r in train_states]))
        rank_x,rank_y=ranker_training(train_episodes); ranker=Logistic().fit(rank_x,rank_y)
        threshold=select_threshold(model,validation_states); thresholds.append(threshold)
        y=np.array([r[3] for r in test_states]); p=model.predict(np.vstack([r[2] for r in test_states])); state_y.extend(y); state_p.extend(p)
        fold_manifest.append({"test":test_date,"validation":sorted(validation_dates),"train":sorted(train_dates),"threshold":threshold})
        for policy in policies: rows.extend(run_policy(policy,test_episodes,model,ranker,threshold,rng))
    detail=pd.DataFrame(rows)
    summary=detail.groupby("policy").agg(episodes=("correct","size"),acceptable_rate=("correct","mean"),abstention_rate=("abstain","mean"),mean_calls=("calls","mean"),mean_cost=("cost","mean"),mean_latency=("latency","mean")).reset_index()
    by_condition=detail.groupby(["policy","condition"]).agg(episodes=("correct","size"),acceptable_rate=("correct","mean"),mean_cost=("cost","mean")).reset_index()
    y=np.asarray(state_y); p=np.asarray(state_p)
    payload={"dataset":"OpenRCA 1.0 Telecom derived locked artifact","action":"return a top-3 root-component triage shortlist","split":"leave-one-telemetry-date-out; two preceding dates calibrate threshold","folds":fold_manifest,"counts":{"incidents":int(len(frame)),"dates":len(dates),"test_episodes":len({(r['case_id'],r['condition']) for r in rows})},"interventions":["canonical","trace_unavailable","stale_metric","wrong_entity_metric","high_trace_cost","temporal_gap"],"sufficiency_model":{"type":"L2 logistic regression","median_threshold":float(np.median(thresholds)),"test_state_AUROC":auc(y,p),"test_state_AUPRC":average_precision(y,p),"test_state_Brier":float(np.mean((p-y)**2)),"test_state_ECE":ece(y,p),"positive_rate":float(y.mean())},"source_model":{"type":"separate L2 logistic source-outcome ranker","training_unit":"masked state plus unobserved source registry entry","selection":"predicted acceptable-action probability minus 0.03 * source cost"},"policy_summary":summary.to_dict("records"),"by_condition":by_condition.to_dict("records"),"limitations":["51 incidents, evaluated with leave-one-date-out folds","source observations are derived summaries rather than raw tool responses","application and middleware sources lack component topology mappings","human probing, authorization, policy sources, and natural source conflicts are absent","action is component top-3 triage, not full component/reason/time RCA"]}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(payload,indent=2)+"\n"); detail.to_csv(output.with_suffix(".csv"),index=False)
    print(summary.to_string(index=False)); print(json.dumps(payload["sufficiency_model"],indent=2)); print(f"wrote {output} and {output.with_suffix('.csv')}")


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--data",type=Path,default=Path("artifacts/openrca_telecom_full_analysis.csv")); parser.add_argument("--output",type=Path,default=Path("artifacts/openrca_context_control_v1.json")); parser.add_argument("--seed",type=int,default=42); args=parser.parse_args(); main(args.data,args.output,args.seed)
