"""Test package.

Present on purpose: without it pytest imports conftest.py as top-level
`conftest` while the test modules' `from tests.conftest import ...` imports it
a second time as `tests.conftest`. Two module objects means two copies of every
class and constant, so `isinstance` checks against fixture-provided objects
fail for no visible reason.
"""
