"""Smoke tests that do not require model weights or GPU."""
from service.app import CLASS_NAMES, valid_confidence
from fastapi import HTTPException


def test_class_count():
    assert len(CLASS_NAMES) == 10
    assert CLASS_NAMES[0] == 'pedestrian'


def test_confidence_bounds():
    valid_confidence(0.25)
    for value in (0.0, 1.01):
        try:
            valid_confidence(value)
        except HTTPException as exc:
            assert exc.status_code == 422
        else:
            raise AssertionError('Expected validation error')
