# Dataset search verdicts

Papers judged by the `/dataset-search` skill, one row per paper, so that a later
sweep does not spend a verification subagent re-answering a settled question.

`decision` is one of `confirmed` (the paper releases a dataset), `rejected` (it
does not), or `unclear` (the verification could not tell). **These are machine
verdicts about whether a dataset exists — they say nothing about licensing.**
Whether a confirmed dataset may actually be redistributed is decided by the
`/dataset-extraction` skill's license stage, and rejections made *by a person*
live in [`rejected_datasets.md`](rejected_datasets.md), not here.

| url | decision | name | record_url | why | date |
|---|---|---|---|---|---|
