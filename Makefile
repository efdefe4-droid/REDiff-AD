.PHONY: check dry-run smoke demo demo-dry-run validate-smoke

RUN_ROOT ?= outputs/smoke_rediff_ad_seed309

check:
	find scripts eval_diversity eval_downstream -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
	python -m pytest -q

dry-run:
	DRY_RUN=1 ANOMALIES_STR=crack SAMPLES_PER_ANOMALY=1 LOG_TO_FILE=0 bash scripts/run_hazelnut_t2r.sh

smoke:
	bash scripts/smoke_hazelnut_t2r.sh

demo:
	bash scripts/run_hazelnut_demo.sh

demo-dry-run:
	DRY_RUN=1 LOG_TO_FILE=0 bash scripts/run_hazelnut_demo.sh

validate-smoke:
	python scripts/validate_smoke_output.py "$(RUN_ROOT)" --defects crack --samples-per-defect 1
