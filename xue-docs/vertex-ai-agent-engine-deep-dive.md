# Building and Hosting Agents with Google Vertex AI Agent Engine — A Deep Technical Guide

> **Based on a thorough analysis of the [`adk-samples`](../) repository (63+ Python agent samples).**

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Writing an Agent: Code Structure & Patterns](#2-writing-an-agent-code-structure--patterns)
3. [Tools: The Agent-to-Tool Protocol](#3-tools-the-agent-to-tool-protocol)
4. [Memory: How Agents Remember](#4-memory-how-agents-remember)
5. [Authentication & Identity](#5-authentication--identity)
6. [Observability, Callbacks, Plugins & Policy](#6-observability-callbacks-plugins--policy)
7. [Testing: Local Development & Validation](#7-testing-local-development--validation)
8. [Deployment: From Code to Production](#8-deployment-from-code-to-production)
9. [Protocol Between Users and Agents](#9-protocol-between-users-and-agents)
10. [Full End-to-End Walkthrough](#10-full-end-to-end-walkthrough)
11. [Summary Table: Key Technologies](#11-summary-table-key-technologies)
12. [Key Hypotheses (Not Directly in Repo)](#12-key-hypotheses-not-directly-in-repo)

---

## 1. Architecture Overview

**Vertex AI Agent Engine** (formerly "Reasoning Engine") is a fully managed Google Cloud service that hosts agents built with the **Agent Development Kit (ADK)**. It abstracts away infrastructure — container management, session persistence, scaling, and model orchestration — so you deploy your `Agent` object and Google runs it as a scalable, stateful API.

### Production Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          USER / CLIENT LAYER                                │
│  Browser  │  Mobile App  │  Backend Service  │  A2A Client  │  CLI         │
└─────────┬───────────────────────────────────────────────────────────────────┘
          │  HTTPS / JSON-RPC (A2A) / gRPC
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GOOGLE CLOUD NETWORKING                                   │
│  Cloud Load Balancer → IAM Auth → Cloud Endpoints                           │
└─────────┬───────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              VERTEX AI AGENT ENGINE (Managed Runtime)                        │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │  AdkApp Container (agent code, serialized via cloudpickle)  │            │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │            │
│  │  │ root_agent   │  │ sub_agent_1  │  │  sub_agent_N     │  │            │
│  │  │ (LlmAgent)   │──│ (LlmAgent)   │──│  (LlmAgent)      │  │            │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────────┘  │            │
│  │         │                 │                  │              │            │
│  │  ┌──────▼───────┐  ┌─────▼────────┐  ┌─────▼───────────┐  │            │
│  │  │ Tools        │  │ Callbacks    │  │ Plugins         │  │            │
│  │  │ (Functions)  │  │ (before/     │  │ (LlmAsAJudge,  │  │            │
│  │  │ (MCP)        │  │  after)      │  │  ModelArmor)    │  │            │
│  │  └──────────────┘  └──────────────┘  └─────────────────┘  │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                                                                             │
│  ┌─────────────────────────┐  ┌────────────────────────────────────┐        │
│  │ SessionService          │  │ MemoryBankService                  │        │
│  │ (VertexAiSessionService)│  │ (VertexAiMemoryBankService)        │        │
│  │  - session CRUD         │  │  - PreloadMemoryTool               │        │
│  │  - state persistence    │  │  - auto_save_session_to_memory     │        │
│  └─────────────────────────┘  └────────────────────────────────────┘        │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │ ArtifactService │ OpenTelemetry Tracing │ Cloud Logging          │       │
│  └──────────────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
          │                    │                         │
          ▼                    ▼                         ▼
┌─────────────────┐  ┌────────────────┐  ┌──────────────────────────┐
│ Gemini Models   │  │ External APIs  │  │ Supporting Services      │
│ (gemini-2.5-    │  │ (Google APIs,  │  │ - Firestore              │
│  flash, pro)    │  │  3rd party)    │  │ - Cloud Storage (GCS)    │
│                 │  │                │  │ - BigQuery               │
│ via Vertex AI   │  │ via Tools /    │  │ - Dataplex               │
│ Generative AI   │  │ MCP Servers    │  │ - Cloud SQL              │
│ API             │  │                │  │ - MCP Toolbox servers    │
└─────────────────┘  └────────────────┘  └──────────────────────────┘
```

---

## 2. Writing an Agent: Code Structure & Patterns

### 2.1 The Core Abstraction: `Agent` / `LlmAgent`

Every agent in this repo inherits from `google.adk.agents.Agent` (or `LlmAgent`). The `Agent` class encapsulates:

- A **model** (which Gemini variant to use)
- An **instruction** (system prompt)
- A list of **tools** (callable functions the LLM can invoke)
- Optional **sub_agents** (for hierarchical multi-agent architectures)
- **Callbacks** at every lifecycle point

**Simple agent** — [`python/agents/customer-service/customer_service/agent.py`](../python/agents/customer-service/customer_service/agent.py):

```python
from google.adk import Agent
from .tools.tools import (
    approve_discount, access_cart_information, modify_cart,
    get_product_recommendations, check_product_availability,
    schedule_planting_service, generate_qr_code, ...
)
from .shared_libraries.callbacks import (
    before_tool, after_tool, before_agent, rate_limit_callback
)

root_agent = Agent(
    model=configs.agent_settings.model,          # e.g. "gemini-2.5-flash"
    global_instruction=GLOBAL_INSTRUCTION,        # shared across all turns
    instruction=INSTRUCTION,                      # agent-specific prompt
    name=configs.agent_settings.name,
    tools=[
        approve_discount, access_cart_information,
        modify_cart, get_product_recommendations,
        check_product_availability, generate_qr_code, ...
    ],
    before_tool_callback=before_tool,             # intercept tool calls
    after_tool_callback=after_tool,               # post-process tool results
    before_agent_callback=before_agent,           # runs before agent starts
    before_model_callback=rate_limit_callback,    # runs before LLM call
)
```

### 2.2 Multi-Agent Hierarchy

**Hierarchical agent** — [`python/agents/travel-concierge/travel_concierge/agent.py`](../python/agents/travel-concierge/travel_concierge/agent.py):

```python
from google.adk.agents import Agent
from travel_concierge.sub_agents.booking.agent import booking_agent
from travel_concierge.sub_agents.in_trip.agent import in_trip_agent
from travel_concierge.sub_agents.inspiration.agent import inspiration_agent
# ... 6 sub-agents total

root_agent = Agent(
    model="gemini-2.0-flash-001",
    name="root_agent",
    description="A Travel Concierge using the services of multiple sub-agents",
    instruction=prompt.ROOT_AGENT_INSTR,
    sub_agents=[
        inspiration_agent, planning_agent, booking_agent,
        pre_trip_agent, in_trip_agent, post_trip_agent,
    ],
    before_agent_callback=_load_precreated_itinerary,
)
```

**How sub-agent delegation works under the hood (hypothesis, based on ADK patterns):** When the root agent's LLM decides it needs a sub-agent, ADK generates a function call with the sub-agent's name. The runtime intercepts this, routes the user message to the sub-agent, and returns its result back into the root agent's conversation. This is conceptually similar to how tools work, but the "tool" is itself an agent with its own LLM, tools, and prompts. The `description` field on each sub-agent is critical — it's injected into the root agent's system prompt so the LLM knows when to delegate.

### 2.3 Standard Project Structure

```
agent_name/
├── .env.example              # Environment variables template
├── README.md                 # Documentation
├── pyproject.toml            # Dependencies (Poetry/uv)
├── agent_name/               # Python package
│   ├── __init__.py
│   ├── agent.py              # Main agent definition (exports `root_agent`)
│   ├── config.py             # Configuration management
│   ├── tools/tools.py        # Tool function definitions
│   ├── prompts/              # Instruction markdown files
│   ├── shared_libraries/
│   │   └── callbacks.py      # Lifecycle callbacks
│   └── memory.py             # (Optional) Custom memory logic
├── deployment/
│   └── deploy.py             # Vertex AI Agent Engine deployment
├── tests/
│   ├── unit/test_tools.py
│   └── integration/test_agent.py
└── uv.lock
```

---

## 3. Tools: The Agent-to-Tool Protocol

### 3.1 How Tools Work (The Protocol)

Tools are **plain Python functions** that the LLM can call. ADK uses the function's **signature + docstring** to auto-generate a JSON Schema that's sent to the Gemini API as a `FunctionDeclaration`. The LLM returns a `function_call` in its response; ADK then invokes the corresponding Python function and feeds the result back as a `function_response`.

**The protocol between agents and tools:**

```
Agent (LLM) ──── function_call{name, args} ────→ ADK Runtime
                                                     │
                                                     ▼
                                              Python Function
                                              (your tool code)
                                                     │
                                                     ▼
Agent (LLM) ←── function_response{result} ──── ADK Runtime
```

This is the Gemini **Function Calling** protocol, which is a structured JSON envelope within the `GenerateContent` API. The key insight: the LLM never directly executes code. It emits a structured `function_call` part, and the runtime (ADK) is the executor.

### 3.2 Defining Tools

**Simple tool** — [`python/agents/customer-service/customer_service/tools/tools.py`](../python/agents/customer-service/customer_service/tools/tools.py):

```python
def approve_discount(
    customer_id: str,
    discount_type: str,
    value: float,
) -> dict:
    """Approves a discount for a customer.

    Args:
        customer_id: The unique customer identifier.
        discount_type: Type of discount ('percentage' or 'fixed').
        value: The discount value.
    """
    MAX_PCT_DISCOUNT = 10
    MAX_FIXED_DISCOUNT = 20

    if discount_type == "percentage" and value > MAX_PCT_DISCOUNT:
        return {
            "status": "rejected",
            "reason": f"Discount exceeds max {MAX_PCT_DISCOUNT}%",
        }
    if discount_type == "fixed" and value > MAX_FIXED_DISCOUNT:
        return {
            "status": "rejected",
            "reason": f"Discount exceeds max ${MAX_FIXED_DISCOUNT}",
        }

    return {"status": "ok", "discount_type": discount_type, "value": value}
```

**The docstring is critical.** ADK parses it to populate the function description sent to Gemini. Type hints become the parameter schema. The LLM sees this schema and decides when/how to call the function.

### 3.3 Tools with `ToolContext` (Stateful Tools)

Tools that need session state receive a `ToolContext` parameter:

```python
from google.adk.tools import ToolContext

def get_user_id(tool_context: ToolContext) -> str | dict[str, str]:
    """Retrieves the user's id to use in the agent's operations."""
    token = tool_context.state.get("auth_token")
    if token:
        return decode_email_from_token(token)
    # Trigger OAuth flow
    tool_context.request_credential(auth_config)
    return {"pending": True, "message": "Awaiting user authentication."}
```

`ToolContext` provides:

- `tool_context.state` — session state dict (shared across tools/turns)
- `tool_context.session` — the current session object
- `tool_context.get_auth_response(config)` — retrieve exchanged OAuth credentials
- `tool_context.request_credential(config)` — trigger OAuth consent flow

### 3.4 MCP Tools (Model Context Protocol)

The repo also uses **MCP (Model Context Protocol)** for tools, enabling tools to be served as separate microservices:

**MCP Toolset** — [`python/agents/currency-agent/currency_agent/agent.py`](../python/agents/currency-agent/currency_agent/agent.py):

```python
from google.adk.tools.mcp_tool import MCPToolset, StreamableHTTPConnectionParams

root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="currency_agent",
    tools=[
        MCPToolset(
            connection_params=StreamableHTTPConnectionParams(
                url=os.getenv("MCP_SERVER_URL", "http://localhost:8080/mcp")
            )
        )
    ],
)
```

**Safe MCP with auth** — [`python/agents/policy-as-code/policy_as_code_agent/mcp.py`](../python/agents/policy-as-code/policy_as_code_agent/mcp.py):

```python
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams

class SafeMCPToolset(McpToolset):
    """Catches connection errors so agent still works if MCP server is down."""

    async def get_tools(self, *args, **kwargs) -> list:
        try:
            return await super().get_tools(*args, **kwargs)
        except Exception as e:
            logging.error(f"Failed to connect to MCP server: {e}")
            return []


def _get_dataplex_mcp_toolset():
    # Get ID token for authenticated MCP calls
    token = id_token.fetch_id_token(auth_req, DATAPLEX_MCP_SERVER_URL)
    return SafeMCPToolset(
        connection_params=SseConnectionParams(
            url=f"{DATAPLEX_MCP_SERVER_URL}/sse",
            headers={"Authorization": f"Bearer {token}"},
        )
    )
```

**The MCP protocol** is a standard for tool communication: the client (ADK) connects to an MCP server via HTTP SSE (Server-Sent Events) or Streamable HTTP, discovers available tools via a schema endpoint, and invokes them via JSON-RPC. This decouples tool implementation from the agent — tools can be in any language, deployed anywhere.

---

## 4. Memory: How Agents Remember

The repo demonstrates a **two-layer memory architecture**:

### 4.1 Layer 1: Conversational Memory (Agent Engine Native)

Agent Engine provides **`VertexAiMemoryBankService`** natively. It stores session turns and enables cross-session recall.

**How it works** — [`python/agents/policy-as-code/policy_as_code_agent/agent.py`](../python/agents/policy-as-code/policy_as_code_agent/agent.py):

```python
from google.adk.tools.preload_memory_tool import PreloadMemoryTool

# Auto-save session to memory after each agent turn
async def auto_save_session_to_memory_callback(callback_context):
    if (
        hasattr(callback_context._invocation_context, "memory_service")
        and callback_context._invocation_context.memory_service
    ):
        await (
            callback_context
            ._invocation_context
            .memory_service
            .add_session_to_memory(
                callback_context._invocation_context.session
            )
        )


root_agent = Agent(
    name="policy_as_code",
    model="gemini-2.5-flash",
    tools=[
        find_policy_in_memory,
        save_policy_to_memory,
        PreloadMemoryTool(),  # Injects relevant past context before LLM call
        # ...
    ],
    after_agent_callback=auto_save_session_to_memory_callback,
)
```

**`PreloadMemoryTool`** is the key abstraction. When the agent starts processing a turn, this tool is automatically invoked to search the memory bank for relevant past interactions and inject them into the LLM's context window. Under the hood (**hypothesis**), it likely does a semantic search (embedding-based) over stored session summaries and prepends the most relevant ones to the system prompt or conversation history.

**`auto_save_session_to_memory_callback`** runs after each agent turn and saves the current session (all turns) to the memory bank. This is how the agent "learns" from conversations.

### 4.2 Layer 2: Procedural/Domain Memory (Custom Firestore)

For structured, long-term, cross-user knowledge, the repo uses **Firestore with vector embeddings**:

**Firestore vector memory** — [`python/agents/policy-as-code/policy_as_code_agent/memory.py`](../python/agents/policy-as-code/policy_as_code_agent/memory.py):

```python
from google.cloud import firestore
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector
from vertexai.language_models import TextEmbeddingModel

db = firestore.Client(project=PROJECT_ID, database=FIRESTORE_DATABASE)


def _get_embedding(text: str) -> list[float]:
    """Generates a vector embedding using Vertex AI text-embedding-004."""
    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
    embeddings = model.get_embeddings([text])
    return embeddings[0].values


def find_policy_in_memory(query: str, source: str, ...) -> dict:
    """Finds a similar policy using Firestore Vector Search."""
    query_embedding = _get_embedding(query)

    vector_query = db.collection(COLLECTION_NAME).find_nearest(
        vector_field="embedding",
        query_vector=Vector(query_embedding),
        distance_measure=DistanceMeasure.COSINE,
        limit=10,
        distance_result_field="similarity_distance",
    )
    results = list(vector_query.stream())
    # Post-filter by source, author, date range
    # Sort by similarity, return best match
    ...


def save_policy_to_memory(
    natural_language_query: str, policy_code: str, ...
) -> dict:
    """Saves a policy with vector embedding for future retrieval."""
    embedding_list = _get_embedding(natural_language_query)
    doc_data = {
        "policy_id": policy_id,
        "version": new_version,
        "query": natural_language_query,
        "embedding": Vector(embedding_list),
        "code": policy_code,
        "created_at": datetime.datetime.now(),
        ...
    }
    db.collection(COLLECTION_NAME).add(doc_data)
```

### Memory Comparison Table

| Feature | Agent Engine Native Memory | Firestore Custom Memory |
|---------|---------------------------|------------------------|
| **Primary Goal** | Maintain chat context across sessions | Store domain knowledge (code, policies) |
| **Technology** | VertexAiMemoryBankService + PreloadMemoryTool | Firestore + Vector Search + text-embedding-004 |
| **Scope** | Per-user, session-based | Global, shared across users |
| **Data Type** | Unstructured text (chat logs) | Structured JSON + vectors + code |
| **Search** | Semantic (automatic) | Vector similarity (cosine distance) |

### 4.3 Simple In-Memory State

For simpler agents, state is just a Python dict in `ToolContext`:

```python
# In tools:
tool_context.state["customer_profile"] = customer.to_json()
cart = tool_context.state.get("cart", [])

# In callbacks:
callback_context.state["timer_start"] = time.time()
callback_context.state["request_count"] = 1
```

When running locally with `InMemorySessionService`, this state lives in RAM. When deployed to Agent Engine, it's automatically persisted by `VertexAiSessionService`.

---

## 5. Authentication & Identity

### 5.1 OAuth2 Flow for User Identity

The brand-aligner agent has the most complete OAuth2 implementation — [`python/agents/brand-aligner/brand_aligner_agent/auth.py`](../python/agents/brand-aligner/brand_aligner_agent/auth.py):

```python
from google.adk.auth.auth_credential import (
    AuthCredential, AuthCredentialTypes, OAuth2Auth,
)
from google.adk.auth.auth_tool import AuthConfig

# 1. Define OAuth2 scheme (OpenAPI standard)
auth_scheme = OAuth2(
    flows=OAuthFlows(
        authorizationCode=OAuthFlowAuthorizationCode(
            authorizationUrl=OAUTH_AUTH_URI_BASE,
            tokenUrl=OAUTH_TOKEN_URI,
            scopes={
                "https://www.googleapis.com/auth/userinfo.email":
                    "See your email",
            },
        )
    )
)

# 2. Provide client credentials
oauth_cred = AuthCredential(
    auth_type=AuthCredentialTypes.OAUTH2,
    oauth2=OAuth2Auth(
        client_id=OAUTH_CLIENT_ID,
        client_secret=OAUTH_CLIENT_SECRET,
    ),
)

# 3. Bundle into AuthConfig
auth_config = AuthConfig(
    auth_scheme=auth_scheme,
    raw_auth_credential=oauth_cred,
)
```

### 5.2 The Auth Flow in Detail

```
User Message ──→ Agent ──→ Tool (get_user_id) ──→ Check state for cached token
                                                       │
                                         ┌─────────────┴──────────────┐
                                         │ Token found?               │
                                    ┌────┴────┐                 ┌─────┴─────┐
                                    │   Yes   │                 │    No     │
                                    ▼         │                 ▼           │
                              Use token       │    tool_context             │
                              directly        │    .get_auth_response()    │
                                              │         │                  │
                                              │    ┌────┴────┐            │
                                              │    │ Got it? │            │
                                              │  ┌─┴──┐  ┌──┴──┐        │
                                              │  │Yes │  │ No  │        │
                                              │  ▼    │  ▼     │        │
                                              │ Build │  tool_context   │
                                              │ Creds │  .request_credential()
                                              │       │         │        │
                                              │       │         ▼        │
                                              │       │  Return pending  │
                                              │       │  → ADK shows     │
                                              │       │    OAuth consent  │
                                              │       │    to user       │
                                              ▼       ▼         │        │
                                         Return user ID         │        │
                                         (encoded email)  ◄─────┘        │
```

In **production mode** (deployed to Agent Engine), the auth token exchange is managed by **Gemini Enterprise** — the agent registration includes OAuth config, and tokens arrive pre-exchanged in `tool_context.state[AUTH_ID]`. In **development mode** (local ADK), the tool itself manages the OAuth redirect flow using `tool_context.request_credential()` and `tool_context.get_auth_response()`.

### 5.3 Service-Level Authentication

For agent-to-cloud-service communication:

```python
# Application Default Credentials (ADC) — used everywhere
# Local: gcloud auth application-default login
# Agent Engine: Automatic via the service account assigned to the engine

import vertexai
vertexai.init(project=PROJECT_ID, location=LOCATION)

# For MCP servers requiring ID tokens:
from google.oauth2 import id_token
token = id_token.fetch_id_token(auth_req, target_audience)
headers = {"Authorization": f"Bearer {token}"}
```

### 5.4 Workload Identity Federation (CI/CD)

The repo's `.github/terraform/main.tf` sets up GitHub OIDC → GCP Workload Identity Federation, allowing GitHub Actions to deploy agents without long-lived service account keys.

---

## 6. Observability, Callbacks, Plugins & Policy

### 6.1 The Callback System (Lifecycle Hooks)

ADK provides hooks at every point in the agent lifecycle:

| Callback | Signature | When it fires |
|----------|-----------|---------------|
| `before_agent_callback` | `(InvocationContext) → None` | Before agent starts processing |
| `before_model_callback` | `(CallbackContext, LlmRequest) → None` | Before LLM API call |
| `after_model_callback` | `(CallbackContext, LlmResponse) → LlmResponse?` | After LLM response |
| `before_tool_callback` | `(BaseTool, args, CallbackContext) → dict?` | Before tool execution |
| `after_tool_callback` | `(BaseTool, args, ToolContext, result) → dict?` | After tool execution |
| `after_agent_callback` | `(CallbackContext) → None` | After agent finishes |

**Rate limiting callback** — [`python/agents/customer-service/customer_service/shared_libraries/callbacks.py`](../python/agents/customer-service/customer_service/shared_libraries/callbacks.py):

```python
def rate_limit_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> None:
    """Implements RPM quota (10 requests per 60 seconds)."""
    now = time.time()
    if "timer_start" not in callback_context.state:
        callback_context.state["timer_start"] = now
        callback_context.state["request_count"] = 1
        return

    request_count = callback_context.state["request_count"] + 1
    elapsed_secs = now - callback_context.state["timer_start"]

    if request_count > RPM_QUOTA:
        delay = RATE_LIMIT_SECS - elapsed_secs + 1
        if delay > 0:
            time.sleep(delay)  # Block until rate limit window resets
        callback_context.state["timer_start"] = now
        callback_context.state["request_count"] = 1
    else:
        callback_context.state["request_count"] = request_count
```

**Tool validation callback** (same file):

```python
def before_tool(tool: BaseTool, args: dict, tool_context: CallbackContext):
    lowercase_value(args)  # Normalize all values

    if "customer_id" in args:
        valid, err = validate_customer_id(
            args["customer_id"], tool_context.state
        )
        if not valid:
            return err  # Short-circuit: return error to LLM instead of calling tool

    if tool.name == "sync_ask_for_approval":
        if args.get("value", None) <= MAX_DISCOUNT_RATE:
            return {
                "status": "approved",
                "message": "No manager needed.",
            }  # Skip tool
```

### 6.2 Cloud Logging Integration

**Structured logging** — [`python/agents/hierarchical-workflow-automation/cookie_scheduler_agent/callback_logging.py`](../python/agents/hierarchical-workflow-automation/cookie_scheduler_agent/callback_logging.py):

```python
import google.cloud.logging

def log_query_to_model(
    callback_context: CallbackContext, llm_request: LlmRequest
):
    cloud_logging_client = google.cloud.logging.Client()
    cloud_logging_client.setup_logging()
    if llm_request.contents and llm_request.contents[-1].role == "user":
        last_user_message = llm_request.contents[-1].parts[0].text
        logging.info(
            f"[query to {callback_context.agent_name}]: "
            + last_user_message
        )


def log_model_response(
    callback_context: CallbackContext, llm_response: LlmResponse
):
    cloud_logging_client = google.cloud.logging.Client()
    cloud_logging_client.setup_logging()
    for part in llm_response.content.parts:
        if part.text:
            logging.info(
                f"[response from {callback_context.agent_name}]: "
                + part.text
            )
        elif part.function_call:
            logging.info(
                f"[function call from {callback_context.agent_name}]: "
                + part.function_call.name
            )
```

### 6.3 OpenTelemetry Tracing

**Arize integration** — [`python/agents/travel-concierge/travel_concierge/tracing.py`](../python/agents/travel-concierge/travel_concierge/tracing.py):

```python
from arize.otel import register
from openinference.instrumentation.google_adk import GoogleADKInstrumentor


def instrument_adk_with_arize() -> trace.Tracer:
    tracer_provider = register(
        space_id=os.getenv("ARIZE_SPACE_ID"),
        api_key=os.getenv("ARIZE_API_KEY"),
        project_name="adk-travel-concierge",
    )
    GoogleADKInstrumentor().instrument(tracer_provider=tracer_provider)
    return tracer_provider.get_tracer(__name__)
```

When deployed to Agent Engine, telemetry can also be enabled natively:

```python
adk_app = agent_engines.AdkApp(agent=root_agent, enable_tracing=True)

# Environment variables for native telemetry:
env_vars = {
    "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY": "true",
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "true",
}
```

> **Hypothesis:** When `enable_tracing=True`, Agent Engine automatically configures an OpenTelemetry exporter that sends traces to Cloud Trace (Google's distributed tracing service). The `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` flag controls whether full prompt/response text is included in trace spans (useful for debugging but potentially a privacy concern).

### 6.4 Safety Plugins

ADK has a **plugin system** that wraps the entire agent lifecycle. Plugins differ from callbacks in that they are reusable, composable objects that can be shared across agents.

**LlmAsAJudge** — [`python/agents/safety-plugins/safety_plugins/plugins/agent_as_a_judge.py`](../python/agents/safety-plugins/safety_plugins/plugins/agent_as_a_judge.py):

```python
from google.adk.plugins import base_plugin


class LlmAsAJudge(BasePlugin):
    """Uses a separate LLM to judge safety of inputs/outputs."""

    def __init__(self, judge_agent=default_jailbreak_safety_agent, ...):
        self._runner = InMemoryRunner(
            agent=self._judge_agent, app_name="judge_app"
        )

    async def _is_unsafe(self, message: str) -> bool:
        """Runs the judge LLM on the given message."""
        author, analysis = await util.run_prompt(
            runner=self._runner, message=message
        )
        return "UNSAFE" in analysis

    async def on_user_message_callback(
        self, invocation_context, user_message
    ):
        if await self._is_unsafe(
            f"<user_message>{user_message.parts[0].text}</user_message>"
        ):
            invocation_context.session.state["is_user_prompt_safe"] = False
            return types.Content(
                role="user",
                parts=[Part(text="Safety filter removed this prompt.")],
            )

    async def before_tool_callback(self, tool, tool_args, tool_context):
        if await self._is_unsafe(
            f"<tool_call>{tool.name}({tool_args})</tool_call>"
        ):
            return {"error": "Unsafe tool inputs."}

    async def after_tool_callback(
        self, tool, tool_args, tool_context, result
    ):
        if await self._is_unsafe(
            f"<tool_output>{result}</tool_output>"
        ):
            return {"error": "Unsafe tool output."}

    async def after_model_callback(
        self, callback_context, llm_response
    ):
        model_output = "\n".join(
            [part.text for part in llm_response.content.parts]
        )
        if await self._is_unsafe(
            f"<model_output>{model_output}</model_output>"
        ):
            return types.Content(
                role="model",
                parts=[
                    Part(text="Safety filter removed this response.")
                ],
            )
```

**ModelArmor** — [`python/agents/safety-plugins/safety_plugins/plugins/model_armor.py`](../python/agents/safety-plugins/safety_plugins/plugins/model_armor.py) uses Google's **Model Armor** service (a separate API at `modelarmor.{location}.rep.googleapis.com`) to detect:

- CSAM content
- Malicious URIs
- Responsible AI (RAI) violations
- Prompt injection / jailbreak attempts
- Sensitive data (SDP) exposure

```python
runner = InMemoryRunner(
    agent=root_agent,
    app_name=APP_NAME,
    plugins=[LlmAsAJudge()],  # or [ModelArmorSafetyFilter()]
)
```

---

## 7. Testing: Local Development & Validation

### 7.1 Unit Testing Tools

**Direct function calls** — [`python/agents/customer-service/tests/unit/test_tools.py`](../python/agents/customer-service/tests/unit/test_tools.py):

```python
def test_approve_discount_over_max():
    result = approve_discount(
        customer_id="C001", discount_type="percentage", value=15
    )
    assert result["status"] == "rejected"
```

### 7.2 Integration Testing with InMemoryRunner

**Simple runner** — [`python/agents/financial-advisor/tests/test_agents.py`](../python/agents/financial-advisor/tests/test_agents.py):

```python
from google.adk.runners import InMemoryRunner


@pytest.mark.asyncio
async def test_happy_path():
    runner = InMemoryRunner(agent=root_agent)
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id="test_user"
    )
    content = UserContent(parts=[Part(text="What stocks should I invest in?")])
    response = ""
    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=content,
    ):
        if event.content.parts and event.content.parts[0].text:
            response = event.content.parts[0].text
    assert "financial" in response.lower()
```

### 7.3 Advanced Multi-Turn Testing with Tool Mocking

**Tool replacement** — [`python/agents/policy-as-code/tests/integration/test_conversational_flow.py`](../python/agents/policy-as-code/tests/integration/test_conversational_flow.py):

```python
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, InMemoryArtifactService


def create_test_agent() -> Agent:
    """Replace real tools with mocks preserving names/docstrings."""
    mocks = {
        "find_policy_in_memory": mock_find_policy_in_memory,
        "generate_policy_code_from_gcs": mock_generate_policy_code_from_gcs,
    }
    new_tools = []
    for tool in original_agent.tools:
        tool_name = getattr(tool, "__name__", None)
        if tool_name in mocks:
            mock = mocks[tool_name]
            mock.__name__ = tool_name  # Preserve name for LLM
            mock.__doc__ = tool.__doc__  # Preserve docstring for schema
            new_tools.append(mock)
        else:
            new_tools.append(tool)
    return Agent(name="test_agent", tools=new_tools, ...)


@pytest.fixture
def runner():
    return Runner(
        app_name="test",
        agent=create_test_agent(),
        session_service=InMemorySessionService(),
        artifact_service=InMemoryArtifactService(),
    )


@pytest.mark.asyncio
async def test_multi_turn_conversation(runner):
    # Turn 1: Greeting
    events = await send_message("Hello", runner)
    assert events
    # Turn 2: Provide source → Agent calls mocked tools
    events = await send_message("Check gs://my-bucket/data.json", runner)
    assert any("policy" in str(e) for e in events)
```

### 7.4 Running Tests

```bash
# Install dependencies
cd python/agents/customer-service
poetry install

# Run unit tests
poetry run pytest tests/unit/ -v

# Run integration tests (requires LLM API access)
poetry run pytest tests/integration/ -v

# Or using the ADK CLI:
adk run agent_name      # Interactive REPL
adk web agent_name      # Web UI for testing
```

---

## 8. Deployment: From Code to Production

### 8.1 Step 1: Build the Agent Package

```bash
cd python/agents/customer-service

# Build a wheel file (required for Agent Engine deployment)
poetry build --format=wheel --output=deployment/
# Produces: deployment/customer_service-0.1.0-py3-none-any.whl
```

### 8.2 Step 2: Deploy to Agent Engine

**Deployment script** — [`python/agents/customer-service/deployment/deploy.py`](../python/agents/customer-service/deployment/deploy.py):

```python
import vertexai
from vertexai import agent_engines
from vertexai.preview.reasoning_engines import AdkApp  # Note: preview API, may move to stable
from customer_service.agent import root_agent

# 1. Initialize Vertex AI
vertexai.init(
    project="my-project",
    location="us-central1",
    staging_bucket="gs://my-project-adk-staging",
)

# 2. Wrap agent in AdkApp
app = AdkApp(agent=root_agent, enable_tracing=True)

# 3. Deploy to Agent Engine
remote_app = agent_engines.create(
    app,
    requirements=["customer_service-0.1.0-py3-none-any.whl"],
    extra_packages=["customer_service-0.1.0-py3-none-any.whl"],
    env_vars={
        "GOOGLE_GENAI_USE_VERTEXAI": "true",
        "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY": "true",
    },
    display_name="Customer Service Agent",
)

print(f"Deployed: {remote_app.resource_name}")
# Output: projects/my-project/locations/us-central1/reasoningEngines/1234567890
```

**What happens during `agent_engines.create()` under the hood (hypothesis):**

1. **Serialization**: Your `root_agent` Python object is serialized using `cloudpickle`. This captures the entire object graph — the agent, its tools, sub-agents, callbacks, everything.
2. **Artifact upload**: The pickled agent + wheel files + requirements are uploaded to the staging GCS bucket.
3. **Container creation**: Agent Engine provisions a container (likely based on a pre-built Python image) that includes your wheel and dependencies.
4. **Service registration**: The engine creates an API endpoint (the `resource_name`) that accepts `create_session`, `stream_query`, and `query` calls.
5. **Backend services**: `VertexAiSessionService` and `VertexAiMemoryBankService` are automatically wired up, replacing your local `InMemorySessionService`.

### 8.3 Step 3: Invoke the Deployed Agent

```python
# Get reference to deployed agent
remote_app = agent_engines.get(
    "projects/my-project/locations/us-central1/reasoningEngines/1234567890"
)

# Create a session
session = remote_app.create_session(user_id="user-123")

# Stream a query
for event in remote_app.stream_query(
    user_id="user-123",
    session_id=session["id"],
    message="Hello! I need help with my order.",
):
    if event.get("content"):
        print(event["content"])
```

### 8.4 Updating & Deleting

```python
# Update existing deployment
remote_app = agent_engines.update(
    resource_name="projects/.../reasoningEngines/1234567890",
    agent_engine=updated_app,
    requirements=[...],
    extra_packages=[...],
)

# Delete
agent_engines.delete(
    resource_name="projects/.../reasoningEngines/1234567890"
)
# or:
remote_agent = agent_engines.get(resource_id)
remote_agent.delete(force=True)
```

### 8.5 Alternative: Cloud Run Deployment

For agents needing custom HTTP endpoints, the repo also shows **Cloud Run + FastAPI** deployment:

**FastAPI wrapper** — [`python/agents/data-science/main.py`](../python/agents/data-science/main.py):

```python
from google.adk.cli.fast_api import get_fast_api_app

app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=web_interface_enabled,
    session_service_uri=session_uri,  # Optional: external session store
)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
    )
```

Deploy with:

```bash
gcloud run deploy data-science-agent \
    --source . \
    --region us-central1 \
    --allow-unauthenticated
```

---

## 9. Protocol Between Users and Agents

### 9.1 The ADK Streaming Protocol

The core user-agent interaction follows an **async event stream**:

```python
async for event in runner.run_async(
    user_id="user-123",
    session_id="session-456",
    new_message=types.UserContent("What's the weather?"),
):
    # event.author: "agent_name" or "tool_name"
    # event.content.parts[]: text, function_call, function_response, etc.
    # event.actions: state mutations, transfers
    if event.content and event.content.parts:
        for part in event.content.parts:
            if part.text:
                print(f"[{event.author}]: {part.text}")
            elif part.function_call:
                print(
                    f"[Tool Call]: {part.function_call.name}"
                    f"({part.function_call.args})"
                )
```

Events represent a **stream of all actions** the agent takes — model responses, tool calls, tool results, sub-agent delegations. The content follows Google's Generative AI `Content` / `Part` schema (same as the Gemini API).

### 9.2 The A2A Protocol (Agent-to-Agent)

For inter-agent communication, the repo uses the **A2A (Agent-to-Agent) protocol** — [`python/agents/currency-agent/`](../python/agents/currency-agent/):

```python
from google.adk.a2a.utils.agent_to_a2a import to_a2a

# Convert any ADK agent to an A2A-compatible server
a2a_app = to_a2a(root_agent, port=10000)
```

**A2A Client** — [`python/agents/currency-agent/currency_agent/test_client.py`](../python/agents/currency-agent/currency_agent/test_client.py):

```python
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    SendMessageRequest, MessageSendParams, Task, TaskState,
)

# 1. Discover agent capabilities via Agent Card
resolver = A2ACardResolver(
    httpx_client=httpx_client, base_url="http://localhost:10000"
)
agent_card = await resolver.get_agent_card()

# 2. Create client from agent card
client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)

# 3. Send message (JSON-RPC)
request = SendMessageRequest(
    id=str(uuid4()),
    params=MessageSendParams(
        message={
            "role": "user",
            "parts": [
                {"kind": "text", "text": "How much is 100 USD in CAD?"}
            ],
            "messageId": uuid4().hex,
        }
    ),
)
response = await client.send_message(request)

# 4. Multi-turn via context_id
if response.root.result.status.state == TaskState.input_required:
    context_id = response.root.result.context_id
    follow_up = SendMessageRequest(
        params=MessageSendParams(
            message={
                "role": "user",
                "parts": [{"kind": "text", "text": "in GBP"}],
                "taskId": task_id,
                "contextId": context_id,
            }
        )
    )
```

**A2A Protocol Details (based on the code + hypothesis):**

- **Discovery**: Each agent exposes an **Agent Card** at a well-known URL (e.g., `/.well-known/agent.json`), describing its capabilities, supported message types, and skills.
- **Transport**: JSON-RPC over HTTP. Messages are sent via `POST /` with standard JSON-RPC envelope.
- **Task model**: Each interaction creates a `Task` with states: `submitted → working → completed` or `input_required` (for multi-turn).
- **Context**: `context_id` enables multi-turn conversations across the same task or related tasks.
- **Streaming**: The protocol supports SSE for streaming responses (via `send_message_streaming`).

---

## 10. Full End-to-End Walkthrough

Here's the complete lifecycle from writing code to production invocation:

### Step 1: Create the project

```bash
mkdir my-agent && cd my-agent
poetry init --name my-agent --python ">=3.11"
poetry add google-adk google-cloud-aiplatform
```

### Step 2: Write the agent

```python
# my_agent/agent.py
from google.adk.agents import Agent


def search_products(query: str, max_results: int = 5) -> dict:
    """Searches the product catalog.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.
    """
    # Your implementation here
    return {"results": [...]}


root_agent = Agent(
    model="gemini-2.5-flash",
    name="shopping_assistant",
    instruction="You are a helpful shopping assistant...",
    tools=[search_products],
)
```

### Step 3: Test locally

```bash
# Interactive CLI
adk run my_agent

# Web UI
adk web my_agent

# Programmatic tests
poetry run pytest tests/ -v
```

### Step 4: Build & Deploy

```bash
# Build wheel
poetry build --format=wheel --output=deployment/

# Deploy (run deploy.py or inline)
python deployment/deploy.py
```

### Step 5: Invoke in production

```python
import vertexai
from vertexai import agent_engines

vertexai.init(project="my-project", location="us-central1")
remote_app = agent_engines.get(
    "projects/.../reasoningEngines/12345"
)

session = remote_app.create_session(user_id="prod-user-001")
for event in remote_app.stream_query(
    user_id="prod-user-001",
    session_id=session["id"],
    message="Find me a red jacket under $100",
):
    print(event)
```

---

## 11. Summary Table: Key Technologies

| Aspect | Technology | Repo Example |
|--------|-----------|--------------|
| **Agent definition** | `google.adk.agents.Agent` | [`customer-service/agent.py`](../python/agents/customer-service/customer_service/agent.py) |
| **Multi-agent** | `sub_agents=[]` | [`travel-concierge/agent.py`](../python/agents/travel-concierge/travel_concierge/agent.py) |
| **Tools** | Plain functions + docstrings | [`customer-service/tools/tools.py`](../python/agents/customer-service/customer_service/tools/tools.py) |
| **MCP Tools** | `MCPToolset` + `StreamableHTTPConnectionParams` | [`currency-agent/agent.py`](../python/agents/currency-agent/currency_agent/agent.py) |
| **Callbacks** | `before_tool`, `after_tool`, `before_agent`, `before_model` | [`customer-service/callbacks.py`](../python/agents/customer-service/customer_service/shared_libraries/callbacks.py) |
| **Safety plugins** | `LlmAsAJudge`, `ModelArmorSafetyFilter` | [`safety-plugins/plugins/`](../python/agents/safety-plugins/safety_plugins/plugins) |
| **Memory (Native)** | `PreloadMemoryTool` + `auto_save_session_to_memory` | [`policy-as-code/agent.py`](../python/agents/policy-as-code/policy_as_code_agent/agent.py) |
| **Memory (Custom)** | Firestore + Vector Search + `text-embedding-004` | [`policy-as-code/memory.py`](../python/agents/policy-as-code/policy_as_code_agent/memory.py) |
| **OAuth2 Auth** | `AuthConfig` + `tool_context.request_credential()` | [`brand-aligner/auth.py`](../python/agents/brand-aligner/brand_aligner_agent/auth.py) |
| **Tracing** | OpenTelemetry + Arize / Cloud Trace | [`travel-concierge/tracing.py`](../python/agents/travel-concierge/travel_concierge/tracing.py) |
| **Cloud Logging** | `google.cloud.logging` in callbacks | [`callback_logging.py`](../python/agents/hierarchical-workflow-automation/cookie_scheduler_agent/callback_logging.py) |
| **Deployment** | `agent_engines.create()` + `AdkApp` | [`customer-service/deploy.py`](../python/agents/customer-service/deployment/deploy.py) |
| **Cloud Run** | `get_fast_api_app()` + Uvicorn | [`data-science/main.py`](../python/agents/data-science/main.py) |
| **A2A Protocol** | `to_a2a()` + `A2AClient` | [`currency-agent/`](../python/agents/currency-agent) |
| **Testing** | `InMemoryRunner` / `Runner` + `InMemorySessionService` | [`policy-as-code/tests/`](../python/agents/policy-as-code/tests) |
| **Local dev** | `adk run` / `adk web` | All agents |

---

## 12. Key Hypotheses (Not Directly in Repo)

> These are inferences based on the code patterns and understanding of the Vertex AI platform. They are **not** directly stated in the repository.

1. **Agent Engine serialization**: `cloudpickle` is used to serialize the entire agent graph. This means your tools, callbacks, and sub-agents must be picklable. Lambda functions and complex closures may cause issues.

2. **Session persistence**: When deployed, `VertexAiSessionService` likely uses Spanner or Bigtable behind the scenes to persist session state, replacing the in-memory dict.

3. **Auto-scaling**: Agent Engine likely runs on Google's internal serverless infrastructure (similar to Cloud Functions or Cloud Run), scaling to zero when idle and auto-scaling under load.

4. **Model routing**: The `model` parameter (e.g., "gemini-2.5-flash") is resolved at runtime to the latest available version in the specified Vertex AI region. The runtime handles model-to-endpoint mapping.

5. **PreloadMemoryTool internals**: This likely generates embeddings of the current user message, searches the memory bank (using ANN search), and injects the top-K relevant past interactions as additional context in the system prompt or as a prefixed user message.

6. **`GOOGLE_GENAI_USE_VERTEXAI=true`**: This flag switches the underlying Google GenAI SDK from the consumer Gemini API (`generativelanguage.googleapis.com`) to the Vertex AI API (`aiplatform.googleapis.com`), which provides enterprise features like VPC-SC, CMEK, and audit logging.

7. **Why wheel files**: Agent Engine needs your code as a distributable package because it installs it in a fresh container. The wheel format ensures reproducible dependency resolution. The `extra_packages` parameter handles uploading non-PyPI local packages alongside pip-installable `requirements`.

---

*This analysis is based on a comprehensive review of all 63+ agent samples, 427 Python files matching ADK patterns, 20+ deployment scripts, and supporting documentation in the `adk-samples` repository.*
