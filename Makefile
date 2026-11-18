# The shared venv lives at the main repo root even when running in a worktree.
SHARED := $(shell dirname $$(git rev-parse --path-format=absolute --git-common-dir))
PY ?= $(SHARED)/.cache/venvs/r4/bin/python

.PHONY: doctor
doctor:  ## device, package versions and teacher model cache
	@$(PY) scripts/gen/doctor.py
