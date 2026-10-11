# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-09

### Added

1. Recording runtime: boundaries with stable keys, per-call noise seeds (`cf.rng()`),
   sync and async agents, nested spans.
2. Plans for replay and counterfactuals: `Replace` and `Resample` interventions,
   recorded-noise reuse, content-addressed matching of calls that moved, strict mode.
3. `blame` (earliest sufficient repair, Wilson intervals; `earliest`, `all` and `ranked`
   strategies) and `bisect` (splicing a failing run with a passing one).
4. Run diff with sequence alignment.
5. Content-addressed SQLite store with hash-chained message lists and built-in redaction
   of keys, tokens, emails, phone numbers and card numbers.
6. HTTP capture for `httpx` and `httpx2` clients, so the current OpenAI and Anthropic SDKs
   work unchanged, including streaming.
7. OpenTelemetry GenAI attributes; OTLP JSON export, collector upload and import.
8. Single-file HTML report: timeline, blame bars, run diff, span inspector.
9. pytest plugin (`cf_record`, `cf_tape`) and contract assertions.
10. FaultBench: three agent families with silent, validator-labelled single faults; an
    LLM-judge baseline; blame ablations.
11. `counterfact` command line: `demo`, `runs`, `show`, `report`, `diff`, `blame`,
    `bisect`, `bench`, `export`.
