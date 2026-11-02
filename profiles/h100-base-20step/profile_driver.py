"""Trace the base stage of one job file."""
import dataclasses, json, sys
from nanochat_cpp import pipeline as pl
job = pl.Job.read(sys.argv[1])
job = dataclasses.replace(job, base_steps=int(sys.argv[2]))
stage = pl.stage_base(job)
print(json.dumps(stage.extra, indent=1, sort_keys=True))
print('profile: checkpoint', stage.checkpoint)
