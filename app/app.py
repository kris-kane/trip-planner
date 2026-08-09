import base64, json
from datetime import date
from urllib.parse import urlparse
from flask import Flask, request, jsonify, render_template_string
import requests, psycopg2
from sentence_transformers import SentenceTransformer
from databricks.sdk import WorkspaceClient

app = Flask(__name__)

w = WorkspaceClient()
secret = w.secrets.get_secret(scope="database", key="lakebase-url")
connection_string = base64.b64decode(secret.value).decode("utf-8")
parsed = urlparse(connection_string)
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_conn():
    return psycopg2.connect(
        host=parsed.hostname, port=parsed.port or 5432, dbname=parsed.path.lstrip("/"),
        user=parsed.username, password=parsed.password, sslmode="require",
    )

def search_destinations(query: str, top_k: int = 3) -> list:
    query_vector = model.encode([query])[0]
    query_str = "[" + ",".join(str(float(x)) for x in query_vector) + "]"
    conn = get_conn(); cursor = conn.cursor()
    cursor.execute("""SELECT destination, title, summary, embedding <=> %s::vector AS distance
                       FROM destination_description_embeddings ORDER BY distance LIMIT %s;""",
                   (query_str, top_k))
    results = [{"destination": r[0], "title": r[1], "summary": r[2], "distance": r[3]} for r in cursor.fetchall()]
    cursor.close(); conn.close()
    return results

def get_weather(destination: str) -> dict:
    geo = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                        params={"name": destination, "count": 1}, timeout=10).json()
    if "results" not in geo or not geo["results"]:
        return {"error": f"Could not find location: {destination}"}
    loc = geo["results"][0]
    w_resp = requests.get("https://api.open-meteo.com/v1/forecast",
                           params={"latitude": loc["latitude"], "longitude": loc["longitude"], "current_weather": "true"}, timeout=10).json()["current_weather"]
    a_resp = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality",
                           params={"latitude": loc["latitude"], "longitude": loc["longitude"], "current": "pm10,uv_index"}, timeout=10).json()["current"]
    return {"destination": destination, "temperature_c": w_resp["temperature"], "windspeed": w_resp["windspeed"],
            "pm10": a_resp.get("pm10"), "uv_index": a_resp.get("uv_index")}

def create_trip(user_name: str, destination: str, start_date: str, end_date: str) -> dict:
    conn = get_conn(); cursor = conn.cursor()
    cursor.execute("SELECT user_id FROM users WHERE name = %s", (user_name,))
    row = cursor.fetchone()
    user_id = row[0] if row else None
    if user_id is None:
        cursor.execute("INSERT INTO users (name) VALUES (%s) RETURNING user_id", (user_name,))
        user_id = cursor.fetchone()[0]
    cursor.execute("INSERT INTO trips (user_id, destination, start_date, end_date) VALUES (%s,%s,%s,%s) RETURNING trip_id",
                   (user_id, destination, start_date, end_date))
    trip_id = cursor.fetchone()[0]
    conn.commit(); cursor.close(); conn.close()
    return {"trip_id": trip_id, "user_id": user_id, "destination": destination}

def add_itinerary_item(trip_id: int, day_number: int, activity: str, notes: str = "") -> dict:
    conn = get_conn(); cursor = conn.cursor()
    cursor.execute("INSERT INTO itinerary_items (trip_id, day_number, activity, notes) VALUES (%s,%s,%s,%s) RETURNING item_id",
                   (trip_id, day_number, activity, notes))
    item_id = cursor.fetchone()[0]
    conn.commit(); cursor.close(); conn.close()
    return {"item_id": item_id, "trip_id": trip_id, "activity": activity}

def add_packing_item(trip_id: int, item_name: str) -> dict:
    conn = get_conn(); cursor = conn.cursor()
    cursor.execute("INSERT INTO packing_items (trip_id, item_name) VALUES (%s,%s) RETURNING item_id", (trip_id, item_name))
    item_id = cursor.fetchone()[0]
    conn.commit(); cursor.close(); conn.close()
    return {"item_id": item_id, "trip_id": trip_id, "item_name": item_name}

tool_functions = {"search_destinations": search_destinations, "get_weather": get_weather,
                   "create_trip": create_trip, "add_itinerary_item": add_itinerary_item, "add_packing_item": add_packing_item}

tool_schemas = [
    {"type": "function", "function": {"name": "search_destinations", "description": "Search for destinations and landmarks matching a description.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "get_weather", "description": "Get current live weather and air quality for a destination.",
     "parameters": {"type": "object", "properties": {"destination": {"type": "string"}}, "required": ["destination"]}}},
    {"type": "function", "function": {"name": "create_trip", "description": "Create a new trip. Dates in YYYY-MM-DD format.",
     "parameters": {"type": "object", "properties": {"user_name": {"type": "string"}, "destination": {"type": "string"},
     "start_date": {"type": "string"}, "end_date": {"type": "string"}}, "required": ["user_name", "destination", "start_date", "end_date"]}}},
    {"type": "function", "function": {"name": "add_itinerary_item", "description": "Add an activity to a trip's itinerary.",
     "parameters": {"type": "object", "properties": {"trip_id": {"type": "integer"}, "day_number": {"type": "integer"},
     "activity": {"type": "string"}, "notes": {"type": "string"}}, "required": ["trip_id", "day_number", "activity"]}}},
    {"type": "function", "function": {"name": "add_packing_item", "description": "Add an item to a trip's packing list.",
     "parameters": {"type": "object", "properties": {"trip_id": {"type": "integer"}, "item_name": {"type": "string"}}, "required": ["trip_id", "item_name"]}}},
]

def call_llm(messages, tools):
    headers = w.config.authenticate()
    headers["Content-Type"] = "application/json"
    resp = requests.post(f"{w.config.host}/serving-endpoints/databricks-meta-llama-3-3-70b-instruct/invocations",
                          headers=headers, json={"messages": messages, "tools": tools, "tool_choice": "auto"}, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]

def run_agent(user_message: str):
    messages = [{"role": "system", "content": f"Today's date is {date.today().isoformat()}."},
                {"role": "user", "content": user_message}]
    for _ in range(5):
        choice = call_llm(messages, tool_schemas)
        if not choice.get("tool_calls"):
            return choice.get("content")
        messages.append({"role": "assistant", "content": choice.get("content") or "", "tool_calls": choice["tool_calls"]})
        for call in choice["tool_calls"]:
            fn_name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"])
            result = tool_functions[fn_name](**args)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": str(result)})
    return "Reached max tool-call iterations."

CHAT_HTML = """
<!DOCTYPE html><html><head><title>Trip Planner Agent</title>
<style>body{font-family:sans-serif;max-width:700px;margin:40px auto;}
#chat{border:1px solid #ccc;padding:10px;height:400px;overflow-y:auto;margin-bottom:10px;}
.msg{margin:8px 0;} .user{color:#333;font-weight:bold;} .agent{color:#0a5;}</style></head>
<body><h2>Trip Planner Agent</h2><div id="chat"></div>
<input id="input" style="width:80%;" placeholder="Ask about a destination or plan a trip...">
<button onclick="send()">Send</button>
<script>
async function send(){
  const input = document.getElementById('input');
  const chat = document.getElementById('chat');
  const text = input.value; if(!text) return;
  chat.innerHTML += `<div class="msg user">You: ${text}</div>`;
  input.value = '';
  const resp = await fetch('/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({message: text})});
  const data = await resp.json();
  chat.innerHTML += `<div class="msg agent">Agent: ${data.response}</div>`;
  chat.scrollTop = chat.scrollHeight;
}
</script></body></html>
"""

@app.route("/")
def index():
    return render_template_string(CHAT_HTML)

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    response = run_agent(user_message)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)