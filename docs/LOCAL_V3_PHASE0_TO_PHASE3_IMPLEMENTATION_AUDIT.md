# Local V3 Phase0→Phase3 Implementation Audit

Implemented in this package:

- Chinese/dialogue data source config: `data/sources/local_v3_zh_dialogue_sources.yaml`
- V3 stage budgets: 1M/5M/20M/100M at seq_len=256
- data ratio: general 60%, dialogue 20%, reasoning 15%, divergent 5%
- chat/instruction/conversation formatters in data pipeline
- Local V3 Phase0→Phase3 runner
- generation and generation smoke scripts
- profiler script
- 4070 V3 phase1 configs and safe config
- tests for Chinese dialogue formatting and V3 stage budgets

Local validation performed here:

```bash
python -m compileall -q src scripts tests
PYTHONPATH=.:src pytest -q
```

Result:

```text
37 passed
```

Runtime GPU validation still must be performed on the user's 4070 machine because this environment cannot access the user's CUDA/checkpoints/data cache.
