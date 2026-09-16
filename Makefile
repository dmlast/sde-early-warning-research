.PHONY: install notebook execute benchmarks report-assets report

install:
	python -m pip install -e .

notebook:
	jupyter lab notebooks/local_sde_early_warning.ipynb

execute:
	mkdir -p .cache/matplotlib .cache/ipython
	MPLCONFIGDIR=.cache/matplotlib XDG_CACHE_HOME=.cache IPYTHONDIR=.cache/ipython \
	jupyter nbconvert --to notebook --execute notebooks/local_sde_early_warning.ipynb \
	  --output local_sde_early_warning.ipynb --output-dir notebooks \
	  --ExecutePreprocessor.timeout=1200

benchmarks:
	PYTHONPATH=src python scripts/run_extended_sde_benchmark.py
	PYTHONPATH=src python scripts/run_nonlinear_sde_benchmark.py

report-assets:
	mkdir -p .cache/matplotlib .cache/fontconfig
	MPLCONFIGDIR=.cache/matplotlib XDG_CACHE_HOME=.cache/fontconfig \
	python scripts/build_report_assets.py

report: report-assets
	mkdir -p .cache/tex
	TEXMFVAR=.cache/tex xelatex -interaction=nonstopmode -halt-on-error \
	  -jobname=local_sde_early_warning_report -output-directory=report report/main.tex
	TEXMFVAR=.cache/tex xelatex -interaction=nonstopmode -halt-on-error \
	  -jobname=local_sde_early_warning_report -output-directory=report report/main.tex
