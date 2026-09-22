"""PR Accountability Predictor.

Tracks what companies promise in investor communications and holds them to it:
extract commitments from transcripts, thread restatements across years, gather
evidence, issue verdicts, and predict which open commitments will be kept.

Layer map (see README.md for the flow diagram):

    config      settings, model routing, cost table
    taxonomy    what counts as a promise -- single source of truth
    domain      the object graph
    store       append-only SQLite, run ledger, result cache, memory
    prompts     versioned prompt registry
    llm         the harness: retries, caching, cost accounting
    ingest      discover -> fetch -> normalise
    context     chunking and context assembly
    extract     promise extraction by character offset
    threading_  restatements grouped into commitment threads
    tools       the verification agent's tool surface
    verify      bounded agent loop + adjudication
    evals       gold set, metrics, champion gate
    outcomes    product telemetry + the predictor
    pipeline    stage orchestration
    api         HTTP service
    cli         command line
"""

__version__ = "0.3.0"
