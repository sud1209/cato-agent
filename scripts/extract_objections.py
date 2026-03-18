import json
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.agent.model_factory import get_model

class ObjectionEntry(BaseModel):
    query: str = Field(description="The customer's objection or concern.")
    answer: str = Field(description="The disarming, empathetic response from the Cato agent.")

class ObjectionList(BaseModel):
    entries: List[ObjectionEntry]

def generate_json_from_text():
    PROJECT_ROOT = Path(__file__).parent.parent
    INPUT_FILE = PROJECT_ROOT / "data" / "cato_john_inputs.txt"
    OUTPUT_FILE = PROJECT_ROOT / "data" / "objection_examples.json"

    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found.")
        return

    llm = get_model(temperature=0)
    structured_llm = llm.with_structured_output(ObjectionList)

    with open(INPUT_FILE, "r") as f:
        raw_text = f.read()

    # 4. Prompt for Extraction
    prompt = f"""
    You are an expert data engineer. I have a transcript of conversations between a 'User' and an 'Agent' (Cato).
    
    TASK:
    Extract every unique objection, concern, or question asked by the User and the corresponding response from the Agent.
    
    STYLE GUIDE:
    Ensure the responses (answers) maintain the 'Cato' style: empathetic, transparent, and disarming.
    
    TRANSCRIPT:
    {raw_text}
    """

    print(f"Processing {INPUT_FILE} using {settings.MODEL_PROVIDER}...")
    
    try:
        result = structured_llm.invoke(prompt)
        
        with open(OUTPUT_FILE, "w") as f:
            json.dump([e.model_dump() for e in result.entries], f, indent=4)
        
        print(f"Successfully generated {len(result.entries)} entries in {OUTPUT_FILE}")

    except Exception as e:
        print(f"An error occurred during extraction: {e}")

if __name__ == "__main__":
    generate_json_from_text()