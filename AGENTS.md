# Project guidance

This repository is the standalone CMPE 249 Track A assignment, separate from ReNe.

- Start with README.md, docs/DESKTOP_SETUP.md, and docs/experiment-plan.md.
- On the GPU desktop inspect results/local/environment.json and smoke-test.json if present. A report from another machine is not proof of readiness here.
- Use the project .venv. Do not install dependencies into a global Python environment.
- Never commit datasets, model weights, secrets, or unfiltered prediction dumps.
- Never invent metrics or treat synthetic smoke tests as evaluation evidence.
- Preserve exact commands, settings, environment versions, dataset splits, and label mappings for experiments.
- Finalize taxonomy/ignore policies before cross-dataset comparisons.
- Record actual completed work and remaining work in docs/experiment-plan.md.
