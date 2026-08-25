"""Operator tooling: snapshot and restore the data a deployment holds.

Ships in the image, unlike tests/ and evaluation/, so a backup can be taken by
a one-off container with the same volumes mounted.
"""
