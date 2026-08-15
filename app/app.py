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

# ---------- Tools ----------

def search_destinations(query: str, top_k: int = 3) -> list:
    query_vector = model.encode([query])[0]
    query_str = "[" + ",".join(str(float(x)) for x in query_vector) + "]"
    conn = get_conn(); cursor = conn.cursor()
    cursor.execute("""SELECT destination, title, summary, embedding <=> %s::vector AS distance
                       FROM destination_description_embeddings ORDER BY distance LIMIT %s;""",
                   (query_str, top_k))
    results = [{"destination": r[0], "title": r[1], "summary": r[2], "distance": round(float(r[3]), 3)} for r in cursor.fetchall()]
    cursor.close(); conn.close()
    return results

def get_weather(destination: str) -> dict:
    try:
        geo = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                            params={"name": destination, "count": 1}, timeout=10)
        geo.raise_for_status()
        geo_data = geo.json()
        if "results" not in geo_data or not geo_data["results"]:
            return {"error": f"Could not find location: {destination}"}
        loc = geo_data["results"][0]

        w_resp = requests.get("https://api.open-meteo.com/v1/forecast",
                               params={"latitude": loc["latitude"], "longitude": loc["longitude"], "current_weather": "true"}, timeout=10)
        w_resp.raise_for_status()
        weather_data = w_resp.json()["current_weather"]

        a_resp = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality",
                               params={"latitude": loc["latitude"], "longitude": loc["longitude"], "current": "pm10,uv_index"}, timeout=10)
        a_resp.raise_for_status()
        air_data = a_resp.json()["current"]

        return {"destination": destination, "temperature_c": weather_data["temperature"], "windspeed": weather_data["windspeed"],
                "pm10": air_data.get("pm10"), "uv_index": air_data.get("uv_index")}
    except requests.RequestException as e:
        return {"error": f"Weather lookup failed: {e}"}

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
    return {"trip_id": trip_id, "user_id": user_id, "destination": destination, "start_date": start_date, "end_date": end_date}

def add_itinerary_item(trip_id: int, day_number: int, activity: str, notes: str = "") -> dict:
    conn = get_conn(); cursor = conn.cursor()
    cursor.execute("INSERT INTO itinerary_items (trip_id, day_number, activity, notes) VALUES (%s,%s,%s,%s) RETURNING item_id",
                   (trip_id, day_number, activity, notes))
    item_id = cursor.fetchone()[0]
    conn.commit(); cursor.close(); conn.close()
    return {"item_id": item_id, "trip_id": trip_id, "day_number": day_number, "activity": activity}

def add_packing_item(trip_id: int, item_name: str) -> dict:
    conn = get_conn(); cursor = conn.cursor()
    cursor.execute("INSERT INTO packing_items (trip_id, item_name) VALUES (%s,%s) RETURNING item_id", (trip_id, item_name))
    item_id = cursor.fetchone()[0]
    conn.commit(); cursor.close(); conn.close()
    return {"item_id": item_id, "trip_id": trip_id, "item_name": item_name}

tool_functions = {"search_destinations": search_destinations, "get_weather": get_weather,
                   "create_trip": create_trip, "add_itinerary_item": add_itinerary_item, "add_packing_item": add_packing_item}

WRITE_TOOLS = {"create_trip", "add_itinerary_item", "add_packing_item"}

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
    """Runs the tool-calling loop. Returns (final_text, list_of_tool_call_events)."""
    messages = [{"role": "system", "content": f"Today's date is {date.today().isoformat()}. Be concise and confirm actions clearly."},
                {"role": "user", "content": user_message}]
    events = []
    for _ in range(5):
        choice = call_llm(messages, tool_schemas)
        if not choice.get("tool_calls"):
            return choice.get("content"), events
        messages.append({"role": "assistant", "content": choice.get("content") or "", "tool_calls": choice["tool_calls"]})
        for call in choice["tool_calls"]:
            fn_name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"])
            try:
                result = tool_functions[fn_name](**args)
            except Exception as e:
                result = {"error": str(e)}
            events.append({"tool": fn_name, "args": args, "result": result, "is_write": fn_name in WRITE_TOOLS})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": str(result)})
    return "Reached max tool-call iterations.", events

# ---------- Routes ----------

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    response_text, events = run_agent(user_message)
    return jsonify({"response": response_text, "events": events})

@app.route("/api/dashboard")
def dashboard():
    conn = get_conn(); cursor = conn.cursor()
    cursor.execute("""SELECT trip_id, destination, start_date, end_date FROM trips ORDER BY trip_id DESC""")
    trips = [{"trip_id": r[0], "destination": r[1], "start_date": str(r[2]), "end_date": str(r[3])} for r in cursor.fetchall()]
    cursor.execute("""SELECT item_id, trip_id, day_number, activity, notes FROM itinerary_items ORDER BY trip_id, day_number""")
    itinerary = [{"item_id": r[0], "trip_id": r[1], "day_number": r[2], "activity": r[3], "notes": r[4]} for r in cursor.fetchall()]
    cursor.execute("""SELECT item_id, trip_id, item_name, packed FROM packing_items ORDER BY trip_id""")
    packing = [{"item_id": r[0], "trip_id": r[1], "item_name": r[2], "packed": r[3]} for r in cursor.fetchall()]
    cursor.close(); conn.close()
    return jsonify({"trips": trips, "itinerary": itinerary, "packing": packing})

@app.route("/")
def index():
    return render_template_string(PAGE_HTML)

PAGE_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Waypoint — Trip Planner Agent</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root{
    --ink:#1B2A4A; --ink-soft:#324569; --paper:#F7F3E9; --paper-dim:#EFE9D8;
    --brass:#B8863B; --sage:#6B8F71; --coral:#C1503A; --line:#D8CFBB;
    --radius:10px;
  }
  *{box-sizing:border-box;}
  body{margin:0; background:var(--ink); color:var(--paper); font-family:'Inter',sans-serif; height:100vh; overflow:hidden;}
  h1,h2,h3,.serif{font-family:'Fraunces',serif;}
  .mono{font-family:'IBM Plex Mono',monospace;}

  .app{display:grid; grid-template-columns: 1.3fr 1fr; height:100vh;}
  @media (max-width: 860px){ .app{grid-template-columns: 1fr;} .dashboard{display:none;} }

  /* ---- Chat panel ---- */
  .chat-panel{display:flex; flex-direction:column; padding:28px 28px 20px; min-height:0;}
  .brand{display:flex; align-items:baseline; gap:10px; margin-bottom:4px;}
  .brand h1{font-size:26px; font-weight:600; margin:0; letter-spacing:0.2px;}
  .brand .compass{width:18px; height:18px; border:1.5px solid var(--brass); border-radius:50%; position:relative; flex-shrink:0;}
  .brand .compass::after{content:''; position:absolute; top:50%; left:50%; width:1px; height:9px; background:var(--brass); transform:translate(-50%,-50%) rotate(35deg);}
  .brand-sub{font-size:12.5px; color:var(--brass); letter-spacing:0.12em; text-transform:uppercase; margin-bottom:22px;}

  .thread{flex:1; overflow-y:auto; padding-right:6px; display:flex; flex-direction:column; gap:14px;}
  .bubble{max-width:88%; padding:12px 16px; border-radius:var(--radius); line-height:1.5; font-size:14.5px;}
  .bubble.user{align-self:flex-end; background:var(--brass); color:var(--ink); border-bottom-right-radius:2px;}
  .bubble.agent{align-self:flex-start; background:rgba(247,243,233,0.06); border:1px solid rgba(247,243,233,0.14); border-bottom-left-radius:2px;}

  .card{background:var(--paper); color:var(--ink); border-radius:8px; padding:12px 14px; margin-top:8px; font-size:13.5px;}
  .card-weather{display:flex; gap:18px; align-items:center;}
  .card-weather .temp{font-family:'Fraunces',serif; font-size:28px; font-weight:600;}
  .card-weather .stats{display:flex; flex-direction:column; gap:2px; color:var(--ink-soft); font-size:12px;}
  .card-search .row{display:flex; justify-content:space-between; gap:10px; padding:6px 0; border-bottom:1px dotted var(--line);}
  .card-search .row:last-child{border-bottom:none;}
  .card-search .title{font-weight:600;}
  .card-search .dist{font-family:'IBM Plex Mono',monospace; font-size:11px; color:var(--brass);}

  .confirm{display:flex; align-items:center; gap:8px; background:rgba(107,143,113,0.15); border:1px solid var(--sage); color:#BFE0C6; padding:8px 12px; border-radius:8px; font-size:13px; margin-top:6px;}
  .confirm::before{content:'✓'; color:var(--sage); font-weight:700;}

  .composer{display:flex; gap:8px; margin-top:16px;}
  .composer input{flex:1; background:rgba(247,243,233,0.08); border:1px solid rgba(247,243,233,0.2); border-radius:8px; padding:12px 14px; color:var(--paper); font-size:14px; font-family:inherit;}
  .composer input::placeholder{color:rgba(247,243,233,0.45);}
  .composer input:focus{outline:2px solid var(--brass); outline-offset:1px;}
  .composer button{background:var(--brass); color:var(--ink); border:none; border-radius:8px; padding:0 20px; font-weight:600; cursor:pointer; font-family:inherit;}
  .composer button:hover{filter:brightness(1.08);}

  /* ---- Dashboard panel ---- */
  .dashboard{background:var(--paper); color:var(--ink); padding:28px 24px; overflow-y:auto; border-left:1px solid rgba(0,0,0,0.06);}
  .dashboard h2{font-size:18px; margin:0 0 4px;}
  .route-divider{border:none; border-top:2px dotted var(--line); margin:18px 0;}
  .trip-stub{background:#fff; border:1px solid var(--line); border-radius:8px; padding:14px 16px; margin-bottom:14px; position:relative;}
  .trip-stub::before, .trip-stub::after{content:''; position:absolute; width:12px; height:12px; background:var(--paper); border-radius:50%; top:50%; transform:translateY(-50%);}
  .trip-stub::before{left:-6px;} .trip-stub::after{right:-6px;}
  .trip-stub .dest{font-family:'Fraunces',serif; font-weight:700; font-size:17px;}
  .trip-stub .dates{font-family:'IBM Plex Mono',monospace; font-size:11.5px; color:var(--brass); margin-top:2px;}
  .trip-stub .section-label{font-size:11px; text-transform:uppercase; letter-spacing:0.08em; color:var(--ink-soft); margin:10px 0 4px;}
  .trip-stub ul{margin:0; padding-left:18px; font-size:13px;}
  .trip-stub li{margin-bottom:3px;}
  .packing li.packed{text-decoration:line-through; color:#999;}
  .empty{color:#8a8474; font-size:13px; font-style:italic;}
</style>
</head>
<body>
<div class="app">
  <div class="chat-panel">
    <div class="brand"><div class="compass"></div><h1>Waypoint</h1></div>
    <div class="brand-sub">Trip planning agent — live weather &amp; destination search</div>
    <div class="thread" id="thread">
      <div class="bubble agent">Ask me to find a destination, check the weather, or plan a trip — e.g. "I want somewhere with ancient temples" or "Plan a trip to Kyoto for me, Oct 1–5."</div>
    </div>
    <div class="composer">
      <input id="input" placeholder="Ask Waypoint…" onkeydown="if(event.key==='Enter') send()">
      <button onclick="send()">Send</button>
    </div>
  </div>
  <div class="dashboard" id="dashboard">
    <h2>Trip Dashboard</h2>
    <div class="brand-sub" style="color:#8a8474;">Live from Lakebase</div>
    <hr class="route-divider">
    <div id="trips-container"><p class="empty">No trips yet — ask the agent to plan one.</p></div>
  </div>
</div>

<script>
async function send(){
  const input = document.getElementById('input');
  const thread = document.getElementById('thread');
  const text = input.value.trim();
  if(!text) return;
  thread.innerHTML += `<div class="bubble user">${escapeHtml(text)}</div>`;
  input.value = '';
  thread.scrollTop = thread.scrollHeight;

  const resp = await fetch('/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({message: text})});
  const data = await resp.json();

  let cardsHtml = '';
  (data.events || []).forEach(ev => {
    if(ev.tool === 'get_weather' && !ev.result.error){
      const r = ev.result;
      cardsHtml += `<div class="card card-weather">
        <div class="temp">${r.temperature_c}°C</div>
        <div class="stats"><div>${r.destination}</div><div>Wind ${r.windspeed} km/h</div><div>UV ${r.uv_index ?? '—'} · PM10 ${r.pm10 ?? '—'}</div></div>
      </div>`;
    }
    if(ev.tool === 'search_destinations' && Array.isArray(ev.result)){
      cardsHtml += `<div class="card card-search">` + ev.result.map(r =>
        `<div class="row"><div><span class="title">${escapeHtml(r.title)}</span><br><span style="color:#555;">${escapeHtml((r.summary||'').slice(0,90))}…</span></div><div class="dist">${r.distance}</div></div>`
      ).join('') + `</div>`;
    }
    if(ev.is_write && !ev.result.error){
      cardsHtml += `<div class="confirm">${writeLabel(ev)}</div>`;
    }
  });

  thread.innerHTML += `<div class="bubble agent">${escapeHtml(data.response || '')}${cardsHtml}</div>`;
  thread.scrollTop = thread.scrollHeight;
  refreshDashboard();
}

function writeLabel(ev){
  if(ev.tool === 'create_trip') return `Trip to ${ev.result.destination} created (#${ev.result.trip_id})`;
  if(ev.tool === 'add_itinerary_item') return `Added "${ev.result.activity}" to day ${ev.result.day_number}`;
  if(ev.tool === 'add_packing_item') return `Added "${ev.result.item_name}" to packing list`;
  return 'Done';
}

function escapeHtml(s){
  const d = document.createElement('div'); d.textContent = s; return d.innerHTML;
}

async function refreshDashboard(){
  const resp = await fetch('/api/dashboard');
  const data = await resp.json();
  const container = document.getElementById('trips-container');
  if(!data.trips.length){ container.innerHTML = '<p class="empty">No trips yet — ask the agent to plan one.</p>'; return; }

  container.innerHTML = data.trips.map(trip => {
    const items = data.itinerary.filter(i => i.trip_id === trip.trip_id);
    const packing = data.packing.filter(p => p.trip_id === trip.trip_id);
    return `<div class="trip-stub">
      <div class="dest">${escapeHtml(trip.destination)}</div>
      <div class="dates">${trip.start_date} → ${trip.end_date}</div>
      <div class="section-label">Itinerary</div>
      ${items.length ? '<ul>' + items.map(i => `<li>Day ${i.day_number}: ${escapeHtml(i.activity)}</li>`).join('') + '</ul>' : '<p class="empty">Nothing planned yet</p>'}
      <div class="section-label">Packing</div>
      ${packing.length ? '<ul class="packing">' + packing.map(p => `<li class="${p.packed ? 'packed' : ''}">${escapeHtml(p.item_name)}</li>`).join('') + '</ul>' : '<p class="empty">Nothing added yet</p>'}
    </div>`;
  }).join('');
}

refreshDashboard();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)