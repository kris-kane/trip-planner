# Waypoint — AI Trip Planning Agent
 
Waypoint is a tool-calling AI agent that plans trips through conversation: it searches destinations by semantic similarity, pulls live weather and air-quality data, and writes itineraries and packing lists to a database — all surfaced through a chat UI with a live-updating trip dashboard.
 
It's built end-to-end on Databricks: a medallion-architecture Spark pipeline feeds a vector search table, an LLM agent orchestrates five tools against that data, and a Flask app exposes it all through a chat interface.

## How it works
**Data pipeline (`notebooks/`):** Five destinations are geocoded, then enriched with live weather/air-quality readings and Wikipedia summaries pulled from the GeoNames API. Bronze tables land the raw API responses as-is; silver tables join and clean them into an analytics-ready shape.
 
**Embeddings (Lakebase):** Silver destination descriptions are embedded with `sentence-transformers` (384-dim) and written to a pgvector-enabled Postgres table, enabling semantic search over "somewhere with ancient temples"-style queries via cosine distance.
 
**Agent (`app/app.py`):** A Flask app runs a tool-calling loop against a Llama 3.3 70B endpoint on Databricks Model Serving. The agent has five tools — one read against the vector table, one read against live weather APIs, and three writes (create a trip, add an itinerary item, add a packing item) against the operational Postgres tables.
 
**Frontend:** A single-page chat UI ("Waypoint") renders tool results as inline cards — weather readouts, destination search results, write confirmations — and a side panel dashboard reads live from Lakebase to show every trip, itinerary, and packing list as they're created.
 
## Why two databases
 
The project deliberately splits OLAP and OLTP:
- **Unity Catalog (Delta)** holds the bronze/silver batch pipeline — good for large-scale, versioned, analytical data.
- **Lakebase (Postgres)** holds the low-latency operational data the agent reads and writes in real time, plus the pgvector embeddings table the agent queries for semantic search.
This mirrors a common production pattern: keep the heavy analytical ETL separate from the fast, transactional path an application actually serves.
 
## Tech stack
 
| Layer | Tools |
|---|---|
| Data ingestion & transformation | PySpark, Unity Catalog (Delta Lake) |
| Vector store / operational DB | Postgres (Databricks Lakebase), pgvector |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| LLM / agent orchestration | Llama 3.3 70B via Databricks Model Serving, OpenAI-style tool calling |
| Backend | Flask, psycopg2 |
| External data | Open-Meteo (geocoding, weather, air quality), GeoNames (Wikipedia summaries) |
| Frontend | Vanilla JS/HTML/CSS (single-file template) |
 
## Agent tools
 
| Tool | Type | Description |
|---|---|---|
| `search_destinations` | Read | Semantic search over destination descriptions via pgvector cosine distance |
| `get_weather` | Read | Live temperature, windspeed, PM10, and UV index for a destination |
| `create_trip` | Write | Creates a user (if new) and a trip record |
| `add_itinerary_item` | Write | Adds a day-by-day activity to a trip |
| `add_packing_item` | Write | Adds an item to a trip's packing list |
 
## Project structure
 
```
trip-planner/
├── README.md
├── ddl.sql                                  # Lakebase (Postgres) schema: embeddings + OLTP tables
├── screenshot.png
├── notebooks/                                # Databricks notebooks — run in order
│   ├── 01_bronze_ingestion.py.py             # Geocoding, weather, and Wikipedia data → bronze Delta tables
│   ├── 02_silver_transformations.py          # Bronze → silver joins and cleaning
│   ├── 03_lakebase_embedding_generation.py   # Embeds silver descriptions → pgvector table
│   └── 04_agent.py                           # Standalone agent notebook (Unity Catalog Function variant)
└── app/
    ├── app.py                                # Flask app: agent loop + chat UI + dashboard API
    ├── app.yaml                              # Databricks Apps entrypoint config
    └── requirements.txt
```
 
## Running it
This project was built to run on Databricks and was built in the Free Edition. 
