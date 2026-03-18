# Cato Agent - Home Equity AI Assistant

A multiagent AI system designed to help homeowners explore home equity investment (HEI) opportunities. Cato is a conversational AI that qualifies users, addresses concerns, and facilitates bookings for home equity consultations.

## Overview

Cato is a prototype agentic AI system built with **LangChain**, **LLMs** (OpenAI/Anthropic/Ollama), and **Redis** for stateful conversations. The system uses an intelligent routing layer that directs user conversations to specialized agents based on intent analysis.

### Key Features

- **Multi-agent Architecture**: Specialized agents handle different aspects of the conversation
- **Intent Classification**: Automatically routes conversations to the appropriate agent
- **State Management**: Persistent session state stored in Redis
- **RAG Integration**: Knowledge base for product information and qualification rules
- **Flexible LLM Support**: Works with OpenAI, Anthropic, or Ollama models
- **Async-First Design**: Built for high-concurrency operations

## Project Structure

```
cato-agent/
├── app/                          # Main application code
│   ├── core/
│   │   ├── agent/               # AI agents
│   │   │   ├── master.py         # Master router agent (intent classification)
│   │   │   ├── qualifier.py      # Qualification assessment agent
│   │   │   ├── objection_handler.py  # Handles user concerns/objections
│   │   │   ├── booking_agent.py  # Handles booking requests
│   │   │   ├── unqualified_agent.py  # Handles users who don't qualify
│   │   │   ├── model_factory.py  # LLM provider factory
│   │   │   └── embeddings_factory.py  # Embedding model factory
│   │   ├── tools/               # Agent tools
│   │   │   ├── property_tool.py  # Property/user profile lookup
│   │   │   └── rag_tool.py       # RAG engine for knowledge base
│   │   └── config.py            # Configuration settings
│   ├── db/
│   │   └── redis_client.py      # Redis state management
│   └── schemas/
│       ├── agent_responses.py   # Pydantic models for agent outputs
│       └── state.py             # CatoState schema
├── data/                         # Test data and knowledge base
│   ├── objection_examples.json  # Example user objections
│   └── cato_john_inputs.txt   # Sample conversation data
├── scripts/                      # Utility scripts
│   ├── cato_demo.py           # Interactive demo
│   ├── test_suite.py            # Test scenarios
│   ├── seed_objections.py       # Load objections into knowledge base
│   ├── seed_mock_db.py          # Initialize mock data
│   └── extract_objections.py    # Extract objections from data
├── docker-compose.yml           # Docker services (app + Redis)
├── .env                         # Environment configuration
└── README.md                    # This file
```

## Architecture

### Agent Flow

```
User Input
    ↓
[Master Agent] - Intent Classification
    ↓
    ├─→ OBJECTION → [Objection Handler] → Response
    ├─→ QUALIFICATION → [Qualifier Agent] → Check FICO/Equity/Liens
    │                      ├─→ Qualified → Continue
    │                      └─→ Unqualified → [Unqualified Agent]
    ├─→ BOOKING → [Booking Agent] → Schedule/Confirm
    ├─→ INFO/GENERAL → [Info Provider] (via Objection Handler)
    └─→ Default → Fallback Greeting
    ↓
[Redis] - Store session state & chat history
```

### Key Agents

| Agent | Purpose | Input | Output |
|-------|---------|-------|--------|
| **Master** | Route to appropriate agent | User message | Intent + routed response |
| **Qualifier** | Check HEI eligibility | User financial data | Qualification status |
| **Objection Handler** | Address concerns | User concern message | Empathetic response |
| **Booking Agent** | Schedule consultations | User booking intent | Confirmation details |
| **Unqualified Agent** | Handle rejections | Disqualification reason | Polite decline message |

### Intent Types

- **GENERAL**: Greetings, social pleasantries, introductions
- **OBJECTION**: Skepticism, scam concerns, trust issues
- **QUALIFICATION**: Financial data (FICO score, equity, debt)
- **BOOKING**: Requests to talk to a person or schedule
- **INFO**: Questions about product, fees, how HEI works

## Setup & Installation

### Prerequisites

- Python 3.9+
- Docker & Docker Compose (for Redis)
- LLM API Key (OpenAI, Anthropic, or local Ollama)

### Quick Start

1. **Clone the repository**
   ```bash
   cd cato-agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # Or if using uv
   uv sync
   ```

3. **Configure environment variables**
   ```bash
   # Edit .env with your API keys
   cp .env.example .env
   ```

   ```env
   # .env
   MODEL_PROVIDER=openai              # Options: openai, anthropic, ollama
   OPENAI_API_KEY=your_key_here
   REDIS_URL=redis://localhost:6379
   ```

4. **Start Redis services**
   ```bash
   docker-compose up -d
   ```
   This starts:
   - `cato_app`: FastAPI application (port 8000)
   - `cato_redis`: Redis Stack (port 6379)
   - `cato_viz`: RedisInsight UI (port 5540)

5. **Run the demo**
   ```bash
   python scripts/cato_demo.py
   ```

   This opens an interactive chat with Cato:
   ```
   --- Cato Vibe Check (Session: vibe-check-abc123) ---
   Type 'exit' to quit.

   You: Hi, I'm interested in learning about home equity
   Cato: Great! I'm Cato from Home.LLC. Let me see if you might be a good fit...
   ```

## Configuration

### Environment Variables (`.env`)

```env
# LLM Provider
MODEL_PROVIDER=openai               # openai | anthropic | ollama
OPENAI_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...

ANTHROPIC_MODEL=claude-3-haiku-20240307
ANTHROPIC_API_KEY=sk-ant-...

OLLAMA_MODEL=llama3.2

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_VECTOR_INDEX=cato_hei_index
PROPERTIES_DB_NAME=properties.db

# Application
DEBUG=False
DEFAULT_SESSION_TTL=3600            # Session timeout in seconds
APP_NAME=Cato Agentic AI
```

### Model Selection

Cato supports multiple LLM providers. Switch providers by changing `MODEL_PROVIDER`:

```python
# config.py handles automatic selection
llm = get_model(temperature=0)  # Uses provider from .env
```

## Usage Examples

### Interactive Demo

```bash
python scripts/cato_demo.py
```

### Test Suite

Run predefined test scenarios:

```bash
python scripts/test_suite.py
```

### Data Preparation

Seed the knowledge base with objection examples:

```bash
python scripts/seed_objections.py      # Load objections into RAG
python scripts/seed_mock_db.py         # Initialize mock properties
python scripts/extract_objections.py   # Extract objections from data files
```

### Programmatic Usage

```python
import asyncio
from app.core.agent.master import run_master_agent

async def chat_with_cato():
    session_id = "user-123"
    user_input = "I have $200k in equity and a 750 FICO score"

    response = await run_master_agent(session_id, user_input)
    print(response)

asyncio.run(chat_with_cato())
```

## API & Integration

### FastAPI Endpoints

When running via Docker:

```bash
# Health check
curl http://localhost:8000/health

# Chat endpoint
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d {
    "session_id": "user-123",
    "message": "Hi Cato!"
  }
```

### Redis State Management

Session state is stored in Redis:

```python
from app.db.redis_client import redis_manager

# Get session state
state = await redis_manager.get_cato_state(session_id)

# Update session state
await redis_manager.update_cato_state(session_id, {
    "user_name": "John",
    "qual_status": "PENDING"
})
```

### Chat History

Conversations are persisted using Redis:

```python
from langchain_community.chat_message_histories import RedisChatMessageHistory

history = RedisChatMessageHistory(
    session_id=f"chat_{session_id}",
    url=settings.REDIS_URL
)

# History is automatically maintained
history.add_user_message("Hello")
history.add_ai_message("Hi there!")
```

## Qualification Flow

Cato uses a qualification process to determine HEI eligibility:

### Qualification Criteria

| Criterion | Requirement | Status |
|-----------|-------------|--------|
| **FICO Score** | ≥ 620 | Required |
| **Home Equity** | Sufficient value | Required |
| **Liens** | Acceptable number | Checked |

### Qualification Statuses

- **QUALIFIED**: Meets all criteria → proceed to booking
- **UNQUALIFIED**: Hard fails (FICO < 620) → polite decline
- **PENDING**: Missing data → request more info

### Disqualification Logic

Hard fails trigger immediate disqualification:

```python
# Hard fail: FICO < 620
if fico_score < 620:
    qual_status = "UNQUALIFIED"
    reason = f"FICO score of {fico_score} is below the 620 minimum"
    # Exit immediately without collecting more data
```

## Data Files

### `data/objection_examples.json`

Sample user objections and concerns:

```json
[
  {
    "objection": "Isn't this a scam?",
    "category": "trust",
    "response": "..."
  },
  ...
]
```

### `data/cato_john_inputs.txt`

Sample conversation for testing and development.

## Testing

### Run Test Suite

```bash
python scripts/test_suite.py
```

The test suite validates:
- Intent classification accuracy
- Agent routing logic
- Qualification rules
- Objection handling
- Booking flow

### Debug Mode

Enable debug logging:

```env
DEBUG=True
```

## Troubleshooting

### Redis Connection Failed

```
Error: Cannot connect to redis://localhost:6379
```

**Solution**: Ensure Redis is running:
```bash
docker-compose up -d redis
docker ps  # Verify containers
```

### LLM API Key Invalid

```
Error: Invalid API key for OpenAI
```

**Solution**: Check your `.env` file:
```bash
echo $OPENAI_API_KEY  # Verify key is set
# Regenerate key from provider dashboard
```

### Session Not Persisting

**Solution**: Verify Redis persistence is enabled and volume is mounted:
```bash
docker-compose logs redis  # Check Redis logs
docker-compose down -v    # Clean volumes if needed
```

## Performance & Scaling

### Current Architecture

- **Concurrency**: Async-first design supports multiple concurrent sessions
- **State Storage**: Redis provides sub-millisecond latency for state lookups
- **Vector Search**: Redis Stack supports semantic search via embeddings
- **Chat History**: Stored in Redis with configurable TTL (default: 1 hour)

### Optimization Tips

1. **Increase Redis memory**:
   ```yaml
   # docker-compose.yml
   environment:
     - REDIS_ARGS=--maxmemory 2gb --maxmemory-policy allkeys-lru
   ```

2. **Batch processing**: Use async agents for parallel tasks

3. **Model selection**: Use faster models (gpt-4o-mini) for qualification, more capable models for objections

## Development

### Project Structure

- **Agents**: Pure LLM logic with structured outputs (Pydantic models)
- **Tools**: Deterministic lookups (properties, rules, knowledge base)
- **Schemas**: Type-safe request/response contracts
- **DB**: Redis abstraction for state and history

### Adding a New Agent

1. Create `app/core/agent/my_agent.py`:
   ```python
   from langchain.agents import create_agent
   from app.core.agent.model_factory import get_model

   llm = get_model(temperature=0)

   MY_PROMPT = "You are..."

   my_agent = create_agent(
       model=llm,
       tools=[...],
       system_prompt=MY_PROMPT
   )
   ```

2. Update master agent routing:
   ```python
   # app/core/agent/master.py
   if "MY_INTENT" in intent:
       result = await my_agent.ainvoke({...})
   ```

### Testing Locally

```bash
# Run demo with your agent
python scripts/cato_demo.py

# Run tests
python scripts/test_suite.py
```

## License

[Add appropriate license information]

## Support & Contact

For questions or issues, refer to the `data/` directory for example inputs or check the test suite in `scripts/test_suite.py`.

---

**Last Updated**: 2026-03-17
