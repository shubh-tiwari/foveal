"""Passive instrumentation: log every image an agent harness sends to a model."""

from foveal.instrument.client import InstrumentedAnthropic, Tracer
from foveal.instrument.records import CallRecord, ImageRef, JsonlSink
from foveal.instrument.tokens import TokenCounter, estimate_image_tokens

__all__ = [
    "CallRecord",
    "ImageRef",
    "InstrumentedAnthropic",
    "JsonlSink",
    "TokenCounter",
    "Tracer",
    "estimate_image_tokens",
]
