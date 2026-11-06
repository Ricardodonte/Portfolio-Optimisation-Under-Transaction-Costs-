# Portfolio Optimization with Transaction Costs

A Python project for portfolio rebalancing with fixed and proportional transaction costs. It reproduces the method of Lobo, Fazel, and Boyd and uses variance curvature to rule out trades before an exact search.

## Run the benchmarks

```bash
python -m pip install -r requirements.txt
python main.py --suite echelle
```

This compares the screening bounds on portfolios of up to 80 assets. Run `python main.py --suite temps` to reproduce the timing comparison, or `python main.py` to run all experiments.

In the 12-asset example, screening reduced the search from 4,096 to 1,024 configurations and runtime from 44.61 to 13.02 seconds. Both methods reached the same optimum within numerical precision.

## Documents

- [English presentation](docs/presentations/Presentation-en.pdf)
- [French presentation](docs/presentations/Presentation-fr.pdf)
- [Original paper](docs/reference/lobo-fazel-boyd-2002.pdf)
