"""A small agent importable by CLI tests (``tests.cli_agent:agent``)."""

import counterfact.runtime as cf


@cf.boundary
def fetch(x: int) -> int:
    return x * 10 + (1 if cf.rng().random() < 0.4 else 0)


def agent() -> int:
    return sum(fetch(x) for x in range(4))


def check(total: int) -> bool:
    return total == 60
