"""Run one config with the extra context methods registered, without changing rac itself.

Usage: python scripts/run_one.py configs/<name>.yaml

oracle_long and oracle_short are the plan's oracle: "the gold snippet first, then the method's
own rule". The registered "oracle" is gold-only, because methods.oracle defaults to fallback="none".
"""

from __future__ import annotations

import functools
import sys

from rac import context
from rac.context import methods
from rac.eval import runner

context.METHODS["oracle_long"] = functools.partial(methods.oracle, fallback="bm25_long")
context.METHODS["oracle_short"] = functools.partial(methods.oracle, fallback="bm25_short")

if __name__ == "__main__":
    print(runner.run(sys.argv[1]), flush=True)
