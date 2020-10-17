.PHONY: clean
clean:
	@echo "Cleaning up ..."
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +

.PHONY: test_pytest
test_pytest:
	@echo "Running pytest ..."
	@pytest -v

.PHONY: test
test: test_pytest

.PHONY: format_black
format_black:
	@echo "Black formatting ..."
	@black .

.PHONY: format_prettier
format_prettier:
	@echo "Prettier formatting ..."
	@@npx prettier --write "**/*.{json,yaml,yml}"

.PHONY: format
format: format_black format_prettier

.PHONY: lint_black
lint_black:
	@echo "Black linting ..."
	@black --check .

.PHONY: lint_prettier
lint_prettier:
	@echo "Prettier linting ..."
	@npx prettier --check "**/*.{json,yaml,yml}"

.PHONY: lint_pylint
lint_pylint:
	@echo "Pylint linting ..."
	@pylint nx_concorde
	@pylint $$(find tests/ -iname "*.py")

.PHONY: lint
lint: lint_black lint_prettier lint_pylint
