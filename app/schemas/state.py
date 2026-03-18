from pydantic import BaseModel, Field
from typing import Optional, Literal

class CatoState(BaseModel):
    session_id: str = ""
    user_name: Optional[str] = None
    
    last_agent: Optional[str] = "Greeter"
    booking_intent: Literal["TRUE", "FALSE", "DID_NOT_PROVIDE"] = "FALSE"
    requal_intent: Literal["TRUE", "FALSE", "DID_NOT_PROVIDE"] = "FALSE"
        
    fico_score: Optional[int] = None
    has_foreclosures: Optional[bool] = None
    has_liens: Optional[bool] = None
    qual_status: Literal["QUALIFIED", "UNQUALIFIED", "PENDING", "UNKNOWN"] = "UNKNOWN"
    
    home_value: Optional[float] = None
    home_equity: Optional[float] = None
    address: Optional[str] = None
