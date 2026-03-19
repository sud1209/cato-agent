import pytest
from app.memory.profile import UserProfile


def test_equity_pct_computed_correctly():
    profile = UserProfile(estimated_value=400_000, mortgage_balance=300_000)
    assert profile.equity_pct == pytest.approx(0.25)


def test_equity_pct_none_when_missing_value():
    profile = UserProfile(mortgage_balance=300_000)
    assert profile.equity_pct is None


def test_equity_pct_none_when_missing_balance():
    profile = UserProfile(estimated_value=400_000)
    assert profile.equity_pct is None


def test_profile_all_none_by_default():
    profile = UserProfile()
    assert profile.name is None
    assert profile.fico_score is None
    assert profile.has_bankruptcy is None
    assert profile.equity_pct is None
