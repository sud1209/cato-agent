from __future__ import annotations
from pydantic import BaseModel, computed_field


class UserProfile(BaseModel):
    name: str | None = None
    property_address: str | None = None
    property_type: str | None = None       # "SFR" | "condo" | "multi-family"
    estimated_value: float | None = None
    mortgage_balance: float | None = None
    fico_score: int | None = None
    has_bankruptcy: bool | None = None

    @computed_field
    @property
    def equity_pct(self) -> float | None:
        if self.estimated_value and self.mortgage_balance:
            return (self.estimated_value - self.mortgage_balance) / self.estimated_value
        return None
