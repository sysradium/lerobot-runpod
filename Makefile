.DEFAULT_GOAL := help

DATASET ?=
POLICY  ?= act
CLOUD   ?= COMMUNITY
TRAIN_ARGS = --dataset "$(DATASET)" --policy $(POLICY) --cloud $(CLOUD) \
	$(if $(STEPS),--steps $(STEPS)) $(if $(BATCH),--batch $(BATCH)) \
	$(if $(MAX_HOURS),--max-hours $(MAX_HOURS)) $(if $(GPU),--gpu "$(GPU)")

.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "  plan/train take DATASET=<id> [POLICY=act] [CLOUD=COMMUNITY] [STEPS=] [BATCH=] [MAX_HOURS=] [GPU=]"

.PHONY: sync
sync: ## Install the dev environment
	uv sync

.PHONY: install
install: ## Install lerobot-runpod as a uv tool
	uv tool install --force --reinstall .

.PHONY: uninstall
uninstall: ## Remove the uv tool
	uv tool uninstall lerobot-runpod

.PHONY: run
run: ## Open the TUI
	uv run lerobot-runpod

.PHONY: test
test: ## Run the test suite
	uv run pytest -q

.PHONY: lint
lint: ## Lint and check formatting
	uv run ruff check .
	uv run ruff format --check .

.PHONY: fmt
fmt: ## Auto-fix lint and format
	uv run ruff check --fix .
	uv run ruff format .

.PHONY: check
check: lint test ## Lint and test (what CI would run)

.PHONY: datasets
datasets: ## List LeLab datasets in the local cache
	uv run lerobot-runpod datasets

.PHONY: gpus
gpus: ## List RunPod GPUs with prices
	uv run lerobot-runpod gpus

.PHONY: pods
pods: ## List lerobot-* pods
	uv run lerobot-runpod pods

.PHONY: plan
plan: _require-dataset ## Dry run: print the plan and pod command, create nothing
	uv run lerobot-runpod train $(TRAIN_ARGS) --dry-run --yes

.PHONY: train
train: _require-dataset ## Train on RunPod (asks for confirmation before creating a pod)
	uv run lerobot-runpod train $(TRAIN_ARGS)

.PHONY: _require-dataset
_require-dataset:
	@test -n "$(DATASET)" || { echo "set DATASET=<owner/name>, see: make datasets" >&2; exit 1; }

.PHONY: clean
clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache build dist
	find . -type d -name __pycache__ -not -path "./.venv/*" -prune -exec rm -rf {} +
