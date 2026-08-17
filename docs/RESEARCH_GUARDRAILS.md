# Context Sufficiency Research Guardrails

This file is the project’s operational north star for an action-conditioned, cost-aware context controller.

## Non-negotiable research question

Every experiment must answer:

> Can an agent decide whether its current evidence is sufficient for a specific action, identify what is missing, acquire valuable context, and act safely at lower cost than broad retrieval?

If an experiment does not include an agent decision, an information-acquisition action, or a comparison of context policies, it is a supporting baseline—not the main research result.

## POMDP completeness check

Before implementation, document:

- hidden world state;
- initial observation/context;
- available sources and source costs;
- acquisition actions;
- returned observations;
- belief/evidence update;
- action gate, abstain, and escalation options;
- reward or utility, including reliability and acquisition cost;
- stopping condition.

If these are not explicit, the work is dataset preparation or an offline baseline, not the context-control experiment.

## Experiment tracks

All work must be labeled as one of:

1. `canonical`: original benchmark performance;
2. `intervention`: paired observation-layer context changes;
3. `controller`: ondemand acquisition and stopping policy;
4. `supporting-baseline`: fixed-context model, dataset audit, feature engineering, or RCA analysis.

Supporting baselines may inform the controller, but cannot replace it or be described as validating the north-star claim.

## Required comparisons

Controller experiments must compare at minimum:

- no additional context;
- always-fetch/all-context;
- fixed top-k retrieval;
- logistic-regression sufficiency gate;
- source ranker without sufficiency gating;
- sufficiency gate plus source ranker;
- oracle source selector where feasible.

Report reliability and efficiency together: acceptable-action rate, unsafe-action rate, safe abstention, calls, tokens, latency, cost, and interaction turns.

## Anti-drift gates

Before accepting a result, answer “yes” to all applicable questions:

- Did the agent choose or control context acquisition?
- Was the underlying world state held fixed across paired conditions?
- Were hidden labels and intervention metadata kept out of observations?
- Were source identity, time, entity scope, provenance, authority, and cost represented?
- Is there a leakage-safe train/validation/test split?
- Was the result compared with an always-fetch policy at equal cost or budget?
- Does the result measure action correctness, not only retrieval or anomaly quality?
- Are stale, missing, conflicting, correlated, and wrong-entity evidence tested?
- Is the conclusion scoped to the tested task and intervention distribution?

Any “no” becomes a stated limitation and blocks a broad research claim.

## Dataset decision rule

Do not select a dataset because it has more telemetry dimensions alone. Select it only if it supports:

- a fixed hidden state;
- controllable observation and source access;
- meaningful acquisition actions;
- action-level ground truth;
- measurable source cost or budget;
- replay or simulation of sequential decisions.

OpenRCA Telecom is currently a `supporting-baseline` and candidate replay environment. It becomes a `controller` benchmark only after modality hiding, query actions, source costs, belief updates, and stopping evaluation are implemented.

## Required experiment record

Every experiment artifact must include:

- north-star subclaim;
- track label;
- POMDP mapping;
- baselines;
- intervention and leakage policy;
- primary reliability metric;
- efficiency metric;
- falsification condition;
- result and scope-limited conclusion;
- explicit next action toward the controller.

## Project status rule

The project is not considered to have advanced the core research merely because a dataset was downloaded, a model improved, or a fixed-context score increased. Core progress requires evidence about the policy that decides what context to acquire and when to stop.
