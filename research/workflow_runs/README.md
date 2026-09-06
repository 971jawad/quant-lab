# Saved workflow runs

Agent outputs are preserved here so a fresh session can recover findings without
re-spending tokens.

## macro-edge-hunt (run `wf_5884517c-d7f`, 2026-09-06)

Five independent analyst lenses proposing macro/policy/news hypotheses, a data
availability scout, then adversarial refutation of every candidate.

- `macro_edge_hunt_journal.jsonl` — one line per completed agent, containing its
  **actual structured return value**. This is the recoverable output; read it
  rather than re-running.
- `macro-edge-hunt-wf_5884517c-d7f.js` — the exact script.

To resume (cached agents return instantly, only new/edited calls re-run):

    Workflow({
      scriptPath: "<path to the .js above, or the copy in this folder>",
      resumeFromRunId: "wf_5884517c-d7f"
    })

To read results directly without any workflow machinery:

    python -c "
    import json
    for l in open('research/workflow_runs/macro_edge_hunt_journal.jsonl'):
        r = json.loads(l)
        if r.get('type') == 'result':
            print(json.dumps(r['result'], indent=2)[:4000])
    "

Context for interpreting them: 735 lagged macro tests had already returned
**fewer nominal hits than chance** (28 vs 37 expected) with **0 of 735 surviving
Benjamini-Hochberg FDR**, so the honest prior on any macro hypothesis here is
low. The lenses were asked to propose only things that are (a) mechanically
testable on free data, (b) genuinely novel versus the failed list, and (c)
economically motivated to PREDICT rather than merely co-move.
