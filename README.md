# 🏘️ Autonomous AI Property Triage Platform

An enterprise-grade, multi-modal microservices platform engineered to automate the intake, multi-step agentic reasoning, computer vision auditing, policy enforcement, and routing of real estate property submissions.

---

## 📌 System Overview

Real estate management platforms face operational bottlenecks processing unstructured incoming property listings and verifying physical asset conditions. This platform implements a resilient, containerized triage pipeline that:
* **Enforces Ingestion & Safety Guardrails:** Restricts non-real-estate submissions, off-topic requests, and adversarial inputs using declarative policy rails.
* **Executes Multi-Modal Computer Vision Auditing:** Dynamically classifies room types and predicts physical property condition scores via a fine-tuned deep learning CNN.
* **Retrieves Semantic Context (RAG):** Contextualizes incoming properties against a persistent vector database of historical listings.
* **Orchestrates Stateful Agent Reasoning:** Coordinates dynamic tool selection and synthesized assessments via an autonomous graph-based agent.
* **Audits Intelligence Outputs:** Verifies generated summaries against hallucinations, false ROI claims, or uninspected legal certifications prior to departmental routing.

---

## 🏗️ System Architecture

The system decouples the user-facing interface, the workflow orchestrator, domain microservices, and local/external inference engines across four distinct layers:

```
[ Layer 1: Streamlit WebUI & Local Conversational Assistant ]
                           │
                           ▼ (HTTP Webhook)
[ Layer 2: n8n Workflow Orchestration Pipeline ]
       ├── 1. Webhook Intake Gateway
       ├── 2. Input Policy Rails (NeMo Guardrails)
       ├── 3. Structured Data Extraction (Google Gemini)
       ├── 4. LangGraph Multi-Tool Agent Execution
       ├── 5. Report Compilation & Markdown Formatting
       ├── 6. Output Policy Rails & Fact Verification
       └── 7. Intent-Based Departmental Routing (Residential / Commercial)
                           │
                           ▼ (Internal Docker Network)
[ Layer 3: Containerized FastAPI Microservices (Docker Compose) ]
       ├── Service 1: RAG Service (Port 8001) ── ChromaDB + LangChain + Llama 3
       ├── Service 2: Image Analyser (Port 8002) ── PyTorch ResNet-50 Dual-Head
       ├── Service 3: Guardrails Engine (Port 8003) ── NeMo Guardrails + Colang
       └── Service 4: LangGraph Agent (Port 8004) ── Stateful Graph Planner & Tools
                           │
                           ▼
[ Layer 4: Model Inference Engines ]
       ├── Local Ollama Server (Meta Llama 3)
       └── External APIs (Google Gemini via REST)
```

---

## 🧩 Microservices Specification

| Service | Port | Endpoint | Core Technologies | Primary Function |
| :--- | :---: | :--- | :--- | :--- |
| **Streamlit WebUI** | `8501` | `/` | Streamlit, Requests | User submission form and local Llama 3 market assistant |
| **n8n Orchestrator** | `5678` | `/webhook/...` | n8n Engine, Node.js | Central pipeline coordination, branching, and routing |
| **RAG Service** | `8001` | `POST /query/` | FastAPI, ChromaDB, LangChain | Persistent vector search and contextual listing retrieval |
| **Image Analyser** | `8002` | `POST /analyse` | FastAPI, PyTorch, Torchvision | Dual-head ResNet-50 room classification & condition scoring |
| **Guardrails Service** | `8003` | `POST /check/input`<br>`POST /check/output` | FastAPI, NeMo Guardrails | Self-checking input and output policy validation |
| **LangGraph Agent** | `8004` | `POST /agent/run` | FastAPI, LangGraph, Pydantic | Stateful multi-step reasoning machine and tool coordination |

---

### Detailed Service Breakdown

#### 1. RAG Service (`rag_service/`)
* **Endpoint:** `POST /query/`
* **Stack:** FastAPI, ChromaDB, LangChain, HuggingFace Sentence Transformers, Ollama (Llama 3).
* **Architecture:** Ingests property descriptions, queries an embedded persistent ChromaDB vector store using Cosine Similarity (`Top-3` retrieval), and formats contextual listing metadata into a constrained prompt template to generate an evidence-grounded recommendation.

#### 2. Image Analyser Service (`image_analyser/`)
* **Endpoint:** `POST /analyse`
* **Stack:** FastAPI, PyTorch, Torchvision, Pillow.
* **Model Architecture:** Custom dual-head network fine-tuned on top of a frozen **ResNet-50** backbone:
  * **Classification Head:** Linear layer outputting class logits across room categories (`kitchen`, `bathroom`, `bedroom`, `living_room`, `exterior`, `other`).
  * **Regression Head:** Linear layer predicting continuous physical condition ratings (1.0 to 5.0).
  * **Confidence Fallback:** Automatically returns an `uncertain` label if the classification softmax probability falls below 0.60.

#### 3. Guardrails Policy Engine (`guardrails_service/`)
* **Endpoints:** `POST /check/input` & `POST /check/output`
* **Stack:** FastAPI, NeMo Guardrails, Colang 1.0, OpenAI-compatible client adapter.
* **Functionality:** Operates a **Self-Checking Rails** architecture. The input gate rejects spam, attacks, and off-topic queries. The output gate inspects compiled reports to prevent hallucinations, fabricated warranties, and unauthorized legal claims.

#### 4. Stateful Reasoning Agent (`langgraph_agent/`)
* **Endpoint:** `POST /agent/run`
* **Stack:** FastAPI, LangGraph, LangChain Core, Pydantic v2.
* **Graph Topology:**
  * **`Planner Node`:** Decomposes complex real estate inquiries to decide whether vector context, visual analysis, or both are required.
  * **`Tool Execution Node`:** Dispatches concurrent HTTP requests across sibling microservices (`rag-service:8001` and `image-service:8002`).
  * **`Synthesiser Node`:** Aggregates multi-modal findings into a unified, evidence-based structural assessment.

---

## 🔄 End-to-End n8n Workflow

The automated n8n pipeline orchestrates the following execution graph:
1. **Webhook Trigger:** Ingests raw property listings and image URLs submitted from the WebUI.
2. **Input Guardrails Check:** Dispatches text to `POST /check/input` on the Guardrails service.
3. **Safety Branch (IF):** Blocks non-compliant submissions immediately; allows verified submissions to proceed.
4. **Information Extractor:** Utilizes an LLM node to extract normalized JSON metadata (property type, city, rooms, square footage).
5. **AI Agent Invocation:** Calls the LangGraph microservice (`POST /agent/run`) to coordinate tool calls and synthesize findings.
6. **Report Compilation (LLM Chain):** Formats extracted features and agent insights into an executive Markdown brief.
7. **Output Policy Check:** Validates the report via `POST /check/output` on the Guardrails service.
8. **Routing Switch:** Routes the validated dossier to the appropriate department (`residential_team` vs. `commercial_team`).

---

## 📊 Current Development Status

- [x] **Service 1 (RAG):** Persistent ChromaDB integration populated with seed listings; Cosine top-3 search verified.
- [x] **Service 2 (Image Analyser):** Dual-head ResNet-50 trained and evaluated with confidence thresholds.
- [x] **Service 3 (Guardrails):** Fail-closed input/output safety policies configured and tested with Colang flows.
- [x] **Service 4 (LangGraph Agent):** Stateful multi-node execution graph with dynamic tool invocation.
- [x] **Workflow Orchestration:** 8-node n8n pipeline verified end-to-end for residential and commercial routing.
- [x] **Containerization:** All microservices packaged with dedicated Dockerfiles and linked via Docker Compose.
- [ ] **Streamlit Interface:** Finalizing dashboard visualization components and real-time report display.
- [ ] **Cloud Deployment:** Production provisioning on AWS EC2 infrastructure.

---

## ⚡ Quickstart & Setup Guide

### 1. Prerequisites
* [Docker Desktop](https://www.docker.com/) installed and running.
* [Ollama](https://ollama.ai/) installed locally with Meta Llama 3:
  ```bash
  ollama pull llama3
  ```

### 2. Environment Configuration
Duplicate the template file and configure required keys:
```bash
cp .env.example .env
```
Ensure your `.env` contains:
```env
GEMINI_API_KEY=your_actual_api_key_here
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
OLLAMA_HOST=http://host.docker.internal:11434
RAG_SERVICE_URL=http://rag-service:8001/query/
IMAGE_SERVICE_URL=http://image-service:8002/analyse
N8N_WEBHOOK_URL=http://localhost:5678/webhook/listing-submission
```

### 3. Launch Services via Docker Compose
Build and run the entire microservices mesh in detached mode:
```bash
docker compose up --build -d
```

### 4. Launch the Web Interface
In your terminal, activate your Python virtual environment and run Streamlit:
```bash
streamlit run app.py
```
* **WebUI:** `http://localhost:8501`
* **n8n Canvas:** `http://localhost:5678`
