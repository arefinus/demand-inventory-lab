# Contributing

This is a small, self-contained lab repository. Contributions are welcome as pull
requests against `main`.

## Ground rules

- Run `make test` before opening a pull request. The suite is offline and CPU-only.
- Do not add real datasets, API keys or cached API responses to the repository.
  Census responses belong in the local cache directory (`.census_cache/`, ignored).
- Do not add reported metrics to README or docs unless a script in this repository
  computed them in the same run and the text says so.
- Keep modules under 300 lines and prefer frozen dataclasses for configuration.
- No `print` in library code; the CLI layer owns all console output.
- Write tests for behaviour, including failure cases (empty series, undefined
  metrics, missing files).

## Development setup

```
python -m pip install -e .
python -m pip install pytest
make smoke
make test
```
