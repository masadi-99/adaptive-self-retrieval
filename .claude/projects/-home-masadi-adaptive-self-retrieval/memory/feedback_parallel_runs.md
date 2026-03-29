---
name: feedback_parallel_runs
description: Run independent experiments in parallel, don't wait for one to finish before starting another
type: feedback
---

Run independent experiments and evaluation runs in parallel using background tasks. Don't wait for one thing to finish before starting another unless they truly depend on each other.

**Why:** User got frustrated watching sequential waits when runs could have been parallelized.

**How to apply:** When running multiple dataset evaluations or testing multiple hypotheses, launch them as parallel background tasks. Only check results when needed for the next decision.
