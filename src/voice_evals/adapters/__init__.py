"""Adapters map a platform payload onto the normalised model.

They are pure functions on purpose: no network, no credentials, no clock. That
is what makes them testable, and an adapter that cannot be tested is an
adapter that silently rots when the platform changes a field name.
"""

from voice_evals.adapters.vapi import call_record_from_vapi

__all__ = ["call_record_from_vapi"]
