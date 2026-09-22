# Judge calibration pairs

Human-labelled `{"verbatim", "claim", "faithful", "note"}` items, as JSON lists.
`python -m pr_predictor eval-judge-check` scores the judge against them and
records the result in `evals/judge_check.json`. Until at least
`config.JUDGE_MIN_CALIBRATION_ITEMS` items agree at `config.JUDGE_MIN_AGREEMENT`,
eval results mark the judge's number `validated: false`.
