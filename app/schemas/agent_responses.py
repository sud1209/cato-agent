from pydantic import BaseModel, Field
from typing import Literal

class QualificationResponse(BaseModel):
    qual_status: Literal["QUALIFIED", "UNQUALIFIED", "PENDING"] = Field(
        description="The final eligibility status of the user."
    )
    reason: str = Field(description="Internal logic/reasoning for the audit trail.")
    message_to_user: str = Field(description="The actual text Cato sends to the user.")