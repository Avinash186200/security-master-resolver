.PHONY: install test bench report demo

install:
	pip install -c constraints.txt -e ".[dev]"

test:
	pytest -q

# Offline: trains on seeds 1-3, scores seed 100 (+101-104), backtest, as-of example.
bench:
	python -m secmaster.cli bench --out results

report:
	python -m secmaster.cli report --out results

# Generate a universe, resolve it into SQLite, query a ticker as of a date.
demo:
	python -m secmaster.cli generate --seed 100 --out data
	python -m secmaster.cli resolve --data data --db out/security_master.sqlite --model results/scorer_combined.json
	python -m secmaster.cli asof --db out/security_master.sqlite --ticker OVSK --date 2012-04-10
	python -m secmaster.cli asof --db out/security_master.sqlite --ticker OVSK --date 2017-01-09
	python -m secmaster.cli history --db out/security_master.sqlite --id R00005
	python -m secmaster.cli history --db out/security_master.sqlite --id R00005 --known-on 2012-08-30
	python -m secmaster.cli asof --db out/security_master.sqlite --ticker ISS --date 2012-08-20 --known-on 2012-08-20
