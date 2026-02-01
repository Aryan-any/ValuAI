# ValuAI - Intelligent Deal Discovery & Pricing Engine

## 📋 Project Overview

**ValuAI** is an advanced AI-powered system that automatically discovers, analyzes, and evaluates online bargains by combining multiple machine learning models with intelligent agent coordination. The system identifies potential deals by scraping real-time product listings from RSS feeds, estimates their true market value using an ensemble of pricing models, and alerts users to significant discounts.

### Core Value Proposition
- **Autonomous Deal Detection**: Continuously scans online marketplaces for potential bargains
- **Multi-Model Pricing**: Leverages 3 different pricing approaches (Fine-tuned LLM, LLM with RAG, Deep Neural Network)
- **Intelligent Filtering**: Only alerts users to deals with discount thresholds > $50
- **Persistent Memory**: Maintains historical deals to avoid duplicate notifications
- **Extensible Architecture**: Modular agent-based design for easy customization and scaling

---

## 🏗️ System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    ValuAI Pipeline Architecture                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. INGESTION LAYER                                            │
│     └─ Deal Scraper (RSS Feeds) → Product Descriptions         │
│                                                                 │
│  2. PROCESSING LAYER                                           │
│     ├─ Scanner Agent (OpenAI) → Validates & Selects Deals      │
│     └─ Preprocessor → Text Normalization (Disabled on Win)     │
│                                                                 │
│  3. VALUATION LAYER                                            │
│     ├─ Frontier Agent (GPT with RAG)                           │
│     ├─ Specialist Agent (Fine-tuned Modal Model)               │
│     └─ Neural Network Agent (Deep NN)                          │
│     └─ Ensemble Agent (Weighted Combination)                   │
│                                                                 │
│  4. DECISION LAYER                                             │
│     ├─ Planning Agent (Orchestration & Filtering)              │
│     └─ Threshold Check ($50 minimum discount)                  │
│                                                                 │
│  5. NOTIFICATION LAYER                                         │
│     └─ Messaging Agent (Pushover Push Notifications)           │
│                                                                 │
│  6. PERSISTENCE LAYER                                          │
│     ├─ Chroma Vector Database (Product Embeddings & Metadata)  │
│     └─ Memory File (memory.json - Deal History)               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Key Components

### 1. **Deal Scraper & Scanner**
- **File**: `agents/deals.py`, `agents/scanner_agent.py`
- **Function**: Fetches product deals from RSS feeds (DealNews)
- **Sources**: Electronics, Computers, Smart Home products
- **Processing**: 
  - Scrapes 10 items per feed (~30 items per cycle)
  - Extracts HTML snippets and cleans text via BeautifulSoup
  - Calls OpenAI GPT-4o-mini with Structured Outputs to filter best 5 deals
  - Only selects deals with valid prices > $0

**Key Classes**:
- `ScrapedDeal`: Raw scraped deal from RSS
- `Deal`: Validated deal with description, price, URL
- `DealSelection`: Top 5 selected deals from a batch

### 2. **Ensemble Pricing Model**
- **File**: `agents/ensemble_agent.py`
- **Purpose**: Combines three independent pricing models for robust estimates
- **Weighting**:
  - Frontier Agent: **80%** (LLM with context from similar products)
  - Specialist Agent: **10%** (Fine-tuned model on Modal)
  - Neural Network Agent: **10%** (Deep learning model)

**Formula**:
$$\text{Final Price} = 0.8 \times \text{Frontier} + 0.1 \times \text{Specialist} + 0.1 \times \text{NeuralNet}$$

### 3. **Frontier Agent (LLM with RAG)**
- **File**: `agents/frontier_agent.py`
- **Model**: OpenAI GPT-4o-mini
- **Method**: 
  1. Encodes product description using SentenceTransformer (`all-MiniLM-L6-v2`)
  2. Queries Chroma vector database for 5 similar products
  3. Uses similar products as context in the prompt
  4. Sends prompt to OpenAI asking for price estimate
  5. Extracts numerical price from response
- **Advantages**: Context-aware pricing using historical products

### 4. **Specialist Agent (Fine-tuned LLM)**
- **File**: `agents/specialist_agent.py`
- **Model**: Custom fine-tuned model hosted on Modal
- **Architecture**: Remote function calling via Modal serverless
- **Purpose**: Specialized pricing model trained on domain-specific pricing patterns
- **Weight**: 10% (secondary validation)

### 5. **Neural Network Agent**
- **File**: `agents/neural_network_agent.py`, `agents/deep_neural_network.py`
- **Architecture**:
  - **Input Layer**: 5000-dimensional HashingVectorizer features
  - **Residual Blocks**: 10 layers of residual connections
  - **Hidden Size**: 4096 units per layer
  - **Dropout**: 0.2 (regularization)
  - **Output Layer**: Single neuron (price prediction)

**Architecture Details**:
```
Input (5000 dims)
    ↓
[Linear → LayerNorm → ReLU → Dropout]
    ↓
[ResidualBlock] × 8 layers
  └─ [Linear → LayerNorm → ReLU → Dropout → Linear → LayerNorm] + skip
    ↓
Output Layer (1 neuron)
    ↓
Denormalization: exp(pred × σ + μ) - 1
    (where μ = 4.435, σ = 1.033)
```

**Features**:
- Text hashing vectorization (5000 features, binary encoding)
- Skip connections for gradient flow
- Layer normalization for training stability
- GPU/CPU/MPS device support

### 6. **Planning Agent (Orchestrator)**
- **File**: `agents/planning_agent.py`
- **Role**: Coordinates all other agents
- **Workflow**:
  1. Calls Scanner Agent to get top 5 deals
  2. For each deal, calls Ensemble Agent to estimate true value
  3. Calculates discount: `discount = estimate - deal_price`
  4. Sorts opportunities by discount (highest first)
  5. Checks if best deal exceeds `$50 threshold`
  6. If threshold met, alerts user via Messaging Agent
  7. Returns highest-value opportunity or None

### 7. **Messaging Agent (Notifications)**
- **File**: `agents/messaging_agent.py`
- **Service**: Pushover API for push notifications
- **Features**:
  - Crafts exciting notification messages using Gemini 2.5 Flash
  - Sends formatted alerts with deal details
  - Sound: "cashregister" notification tone

### 8. **Autonomous Planning Agent (Alternative)**
- **File**: `agents/autonomous_planning_agent.py`
- **Purpose**: LLM with tool-use capabilities
- **Tools Provided**:
  - `scan_the_internet_for_bargains()`: Returns deal list
  - `estimate_true_value()`: Estimates product value
  - `notify_user_of_deal()`: Sends user alert
- **Flow**: Agentic loop with OpenAI function calling

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API Key
- Hugging Face Token (for `sentence-transformers`)
- Google AI API Key (for `Gemini`)

### Installation

1. **Clone the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.base.txt -r requirements.ml.txt
   ```
3. **Configure Environment**:
   - Create a `.env` file (see `.env.example`)
   - Add your API keys

### Required Model File

The system requires a pre-trained Neural Network model to operate at full capacity:
- **File**: `deep_neural_network.pth`
- **Placement**: Root directory
- **Note**: The system is designed to degrade gracefully. If this file is missing, the ensemble will proceed using only the Frontier (and Specialist) agents.

### Running the Application

**Direct Execution**:
```bash
python service.py
```

**API Server**:
```bash
uvicorn api:app --reload
```

### Input → Processing → Output

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPLETE DATA FLOW                           │
└─────────────────────────────────────────────────────────────────┘

INPUT LAYER:
  ↓
  RSS Feeds (DealNews)
  ├─ Electronics feed
  ├─ Computers feed
  └─ Smart Home feed
  
  ↓
SCRAPING:
  └─ BeautifulSoup extracts: title, description, price, URL, features
  
  ↓
VALIDATION:
  └─ Scanner Agent (GPT-4o-mini + Structured Outputs)
     ├─ Validates price > 0
     ├─ Checks for "X% off" or "reduced by" (resolves to actual price)
     ├─ Selects top 5 by description quality
     └─ Returns: Deal(description, price, url)
  
  ↓
MEMORY CHECK:
  └─ Compares URLs against memory.json
     └─ Only processes new deals
  
  ↓
FEATURE EXTRACTION:
  └─ Text preprocessing (currently disabled on Windows)
     └─ Returns: normalized product description
  
  ↓
VALUATION (3-MODEL ENSEMBLE):

  Model 1: Frontier Agent (80% weight)
  ├─ Encode description → SentenceTransformer vector
  ├─ Vector search in Chroma DB → find 5 similar products
  ├─ Build context with historical prices
  ├─ Call GPT-5.1 with context
  └─ Extract price → Frontier prediction
  
  Model 2: Specialist Agent (10% weight)
  ├─ Send description to Modal fine-tuned model
  └─ Extract price → Specialist prediction
  
  Model 3: Neural Network Agent (10% weight)
  ├─ Hash vectorize description (5000 features)
  ├─ Forward pass through 10-layer residual network
  ├─ Denormalize output (exp(z × σ + μ) - 1)
  └─ Extract price → NN prediction
  
  ↓
ENSEMBLE COMBINATION:
  └─ Final Price = 0.8×F + 0.1×S + 0.1×N
  
  ↓
OPPORTUNITY IDENTIFICATION:
  └─ discount = final_estimate - deal_price
     ├─ If discount > $50 → OPPORTUNITY
     └─ If discount ≤ $50 → DISCARD
  
  ↓
ALERTING:
  └─ Messaging Agent
     ├─ Uses Gemini to craft exciting message
     ├─ Sends via Pushover API
     └─ Message format: "{Sentiment} Price=${price}, Discount=${discount} - {URL}"
  
  ↓
PERSISTENCE:
  └─ Memory Update
     ├─ Appends Opportunity to memory.json
     └─ Maintains deal history for deduplication
  
  ↓
OUTPUT:
  └─ List[Opportunity]
     ├─ deal: Deal
     ├─ estimate: float
     ├─ discount: float
```

---

## 🤖 Inference Pipeline (Detailed)

### Single Deal Processing

```python
# Input: Product description (string)
# Output: Opportunity or None

deal_description = "55-inch 4K Smart TV with HDR..."
deal_price = 178.00

# Step 1: Ensemble Agent processes
estimate = ensemble.price(deal_description)
#  ├─ estimate = (frontier_price × 0.8) 
#  │            + (specialist_price × 0.1) 
#  │            + (nn_price × 0.1)
#  └─ Example: estimate = (250 × 0.8) + (240 × 0.1) + (235 × 0.1) = $247.50

# Step 2: Calculate discount
discount = estimate - deal_price  # $247.50 - $178.00 = $69.50

# Step 3: Filter by threshold
if discount > DEAL_THRESHOLD ($50):
    opportunity = Opportunity(
        deal=deal,
        estimate=estimate,
        discount=discount
    )
    # Step 4: Send notification
    messenger.notify(deal_description, deal_price, estimate, url)
    return opportunity
else:
    return None
```

### Model-Specific Inference Details

**Frontier Agent (GPT with RAG)**:
```
description → SentenceTransformer encoder → 384-dim vector
                  ↓
            Vector DB search (Chroma)
                  ↓
          5 similar products + prices
                  ↓
       Build context prompt
                  ↓
    OpenAI Chat Completion (GPT-5.1)
                  ↓
    Extract price from response
```

**Neural Network Agent**:
```
description → HashingVectorizer (5000 features, binary)
                  ↓
         Convert to PyTorch tensor
                  ↓
    Forward pass through 10 layers:
    ├─ Input layer: Linear(5000→4096) + LayerNorm + ReLU + Dropout
    ├─ 8× Residual blocks (same hidden size)
    └─ Output layer: Linear(4096→1)
                  ↓
     Denormalize: exp(output × 1.033 + 4.435) - 1
                  ↓
    Return max(0, price_estimate)
```

---

## 🗄️ Data Persistence

### Chroma Vector Database
- **Path**: `products_vectorstore/`
- **Collection**: "products"
- **Stored Data**:
  - **embeddings**: 384-dim vectors (from SentenceTransformer)
  - **documents**: Product descriptions
  - **metadatas**: `{"price": float, "category": str}`

### Memory File
- **Path**: `memory.json`
- **Format**: JSON array of Opportunity objects
- **Purpose**: 
  - Prevent duplicate deal notifications
  - Maintain historical deal data
  - Enable URL-based deduplication

**Structure**:
```json
[
  {
    "deal": {
      "product_description": "...",
      "price": 178.0,
      "url": "..."
    },
    "estimate": 247.5,
    "discount": 69.5
  }
]
```

---

## 🚀 Execution Flow

### Entry Points

#### 1. **Service Entrypoint** (`service.py`)
```python
def run_pricer_cycle() -> List[Opportunity]:
    framework = get_agent_framework()
    opportunities = framework.run()
    return opportunities
```

#### 2. **API Entrypoint** (`api.py`)
```python
@app.post("/run", response_model=List[Opportunity])
def run_agents():
    return run_pricer_cycle()
```

#### 3. **Direct Execution**
```bash
python deal_agent_framework.py
```

### Execution Steps

1. **Initialization**
   - Load `.env` for API keys
   - Initialize Chroma client
   - Read memory.json
   - Setup Planning Agent with Chroma collection

2. **Deal Scanning**
   - ScannerAgent fetches from RSS feeds
   - BeautifulSoup extracts product data
   - OpenAI validates and selects top 5
   - Filters out URLs already in memory

3. **Pricing & Evaluation**
   - EnsembleAgent processes each deal
   - 3 models run in parallel (conceptually)
   - Weighted combination produces final estimate
   - Discount calculated

4. **Filtering & Alerting**
   - PlanningAgent checks $50 threshold
   - Best deal selected (highest discount)
   - MessagingAgent sends notification if qualified
   - Opportunity appended to memory

5. **Persistence**
   - memory.json updated with new opportunities
   - Chroma vectors indexed

---

## 📚 Models & Algorithms

### Model Summary

| Agent | Model | Input | Output | Weight |
|-------|-------|-------|--------|--------|
| **Frontier** | GPT-5.1 | Description + Similar products | Price estimate | 80% |
| **Specialist** | Fine-tuned Modal | Description | Price estimate | 10% |
| **Neural Net** | ResidualNet (10 layers) | 5000-dim features | Price estimate | 10% |

### Feature Engineering

**Neural Network Features**:
- HashingVectorizer: Binary term frequency
- Dimensions: 5000
- Stop words: English
- Purpose: Fast, deterministic feature extraction

**Frontier Agent Embeddings**:
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimensions: 384
- Purpose: Semantic similarity for RAG

---

## 🔧 Configuration

### Environment Variables (`.env`)

```env
# LLM Providers
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
GEMINI_API_KEY=AIzaSy...

# Notifications
PUSHOVER_USER=...
PUSHOVER_TOKEN=...

# Runtime
ENV=production
```

### Thresholds & Hyperparameters

| Parameter | Value | File |
|-----------|-------|------|
| Deal Threshold | $50 | `planning_agent.py` |
| Ensemble Weights | [0.8, 0.1, 0.1] | `ensemble_agent.py` |
| NN Hidden Size | 4096 | `deep_neural_network.py` |
| NN Layers | 10 | `deep_neural_network.py` |
| NN Dropout | 0.2 | `deep_neural_network.py` |
| Vector DB Results | 5 | `frontier_agent.py` |
| Scanner Deals | 5 | `scanner_agent.py` |

---

## 📦 Dependencies

### Core Stack
- **FastAPI**: REST API framework
- **PyTorch**: Deep learning
- **Transformers**: Hugging Face models
- **SentenceTransformers**: Semantic embeddings
- **Chroma**: Vector database
- **OpenAI**: GPT API
- **LiteLLM**: LLM abstraction layer
- **BeautifulSoup4**: HTML parsing
- **Feedparser**: RSS parsing

### Detailed Dependencies
See `requirements.base.txt`, `requirements.ml.txt`, `requirements.docker.txt`

---

## 🐳 Deployment

### Docker Containerization

**Build**:
```bash
docker build -t valuai:latest .
```

**Run**:
```bash
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e HF_TOKEN=$HF_TOKEN \
  valuai:latest
```

**Healthcheck**:
```
GET /health → {"status": "ok"}
```

---

## 🧪 Testing & Evaluation

### Evaluator Module (`agents/evaluator.py`)

Provides comprehensive performance metrics:
- **Mean Absolute Error (MAE)**: Average prediction error in dollars
- **Mean Squared Error (MSE)**: Penalizes large errors
- **R² Score**: Variance explained by model (0-100%)
- **Error Trends**: Running average error with 95% confidence intervals

**Usage**:
```python
from agents.evaluator import evaluate
evaluate(predictor_function, test_data, size=200)
```

---

## 🔄 Workflow Summary

### Daily/Hourly Cycle

1. **Scan** → Fetch top 5 new deals from RSS
2. **Validate** → Use OpenAI to filter by quality
3. **Price** → Ensemble of 3 models estimates true value
4. **Compare** → Check if discount exceeds $50 threshold
5. **Alert** → Send push notification to user
6. **Remember** → Store opportunity in memory

### Benefits

✅ **Automated**: Runs continuously without manual intervention
✅ **Ensemble**: Combines multiple models for robustness
✅ **Fast**: Parallel potential (with async improvements)
✅ **Scalable**: Vector DB enables efficient similarity search
✅ **Intelligent**: Uses LLMs for semantic understanding
✅ **Persistent**: Never shows same deal twice

---

## 🛠️ Future Enhancements

- Async model execution for parallel pricing
- Fine-tuning ensemble weights based on historical accuracy
- Expand RSS feed sources
- Category-specific pricing models
- Web scraping beyond RSS feeds
- ML model explainability (SHAP values)
- A/B testing notification strategies
- User preference learning

---

## 📞 Support

For issues or questions, refer to:
- `.env.example` for configuration
- Individual agent docstrings for implementation details
- `deep_neural_network.py` for model architecture

---

**ValuAI** © 2026 | AI-Powered Deal Discovery Engine
