# 08 · API & Serving

Every network surface: the REST/WebSocket/gRPC control planes for the discrete
pipeline, plus the standalone FastAPI dashboards (Kundali, Resonance, Decoupled).

Modules: `serving/{broadcast,api,app,binary,grpc_server,webui}.py`,
`proto/kalachakra.proto`, and `scripts/serve*.py`.

---

## 1. `serving/broadcast.py` — the broadcast engine

`BroadcastEngine` serves per‑coordinate topological metrics for a fixed timeline
frame over the 122,880‑node mesh:

- `nearest_node(lat, lon)` — mesh node nearest a query point.
- `query(lat, lon) → LocalReading` — `{potential_index, shear_velocity, cluster_id, …}`.
- `heatmap()` — the full per‑node field for a WebGL upload.

`LocalReading` is the broadcast payload for one point at one frame.

## 2. `serving/api.py` — minimal REST (`/potential`, `/heatmap`)

`create_app(engine, frame=0)` (FastAPI):

| Method | Path | Returns |
|---|---|---|
| GET | `/potential?lat=&lon=` | `PotentialResponse` = `{potential_index, shear_velocity, cluster_id, frame}` |
| GET | `/heatmap` | full per‑node field array |

`PotentialResponse.from_reading(reading, frame)` / `.to_dict()`.

## 3. `serving/app.py` — the radar control plane

`create_app(store_root)` — the full kinetic‑radar API over the tokenized Parquet
index (needs `[index,serve]`):

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | store coverage summary |
| POST | `/inspect` | tier‑routed rows for a viewport + HDBSCAN manifold clusters |
| POST | `/news` | rarest‑first textless geometric news cards |
| POST | `/microgrid` | on‑the‑fly continuous‑LOD regional field (§4 micro‑grid) |
| POST | `/telemetry` | Sidebar‑Inspector telemetry for a coordinate |
| WS | `/stream` | binary field frames (WebSocket) |

Helpers: `_cluster_rows`, `_applying_flag`, `_lookup_stored`. `fastapi_available()`
guards the optional dep. The `/inspect`, `/news`, `/microgrid`, `/telemetry` request
models are Pydantic (`InspectRequest`, `NewsRequest`, …). Served by
`scripts/serve_radar.py`, which also `mount_web_ui`’s `web/` at `/ui`
(`http://host:port/ui/radar.html`).

## 4. `serving/binary.py` — binary frame packing

`pack_frame(lat, lng, potential, shear, macro, micro, …) → bytes` and
`unpack_frame(buf)` — the compact wire format for `/stream` (typed arrays a WebGL/JS
client uploads directly, no JSON parse).

## 5. `serving/grpc_server.py` + `proto/kalachakra.proto` — typed gRPC

The strongly‑typed counterpart to the REST app. Service **`CosmicWeather`**:

```proto
service CosmicWeather {
  rpc Health(HealthRequest)     returns (HealthReply);
  rpc Inspect(InspectRequest)   returns (InspectReply);
  rpc Telemetry(TelemetryRequest) returns (TelemetryReply);
}
```

`make_servicer(store_root)`, `serve(store_root, host, port=50051, max_workers=8)`,
`grpc_available()`. Generated stubs live in `serving/grpc_gen/` (excluded from lint;
regenerate with `[grpc]`). Run: `scripts/serve_grpc.py --index data/index --port 50051`.

## 6. `serving/webui.py` — shared UI plumbing

`enable_cors(app)` (permissive CORS so a `file://` page can call the API),
`default_web_dir()`, `mount_web_ui(app, web_dir, path="/ui")` — mounts `web/` as
static files.

---

## 7. Standalone dashboard servers (`scripts/serve_*.py`)

Each is a self‑contained FastAPI app serving one `web/*.html` dashboard plus its API.

### 7.1 Kundali — `scripts/serve_kundali.py` → `web/kundali.html`

Sidereal twin search over the daily DuckDB (see [10](10-kundali-engine.md)).

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | the Kundali dashboard |
| GET | `/health` | DB coverage + tier names |
| GET | `/api/coastlines.geojson` | world outline |
| POST | `/api/search` | `{date,time,tz_hours,lat,lon,tier,limit,active_planets,active_houses,start_year,end_year}` → natal chart + tier hits + per‑tier availability + `location_free`/`coverage` |

### 7.2 Resonance — `scripts/serve_resonance.py` → `web/resonance.html`

| Method | Path |
|---|---|
| GET | `/`, `/health`, `/api/topology`, `/api/coastlines`, `/api/mesh` |
| POST | `/api/resonance`, `/api/stream_resonance` |

A 3‑D topology/resonance viewer (three.js) over the mesh.

### 7.3 Radar — `scripts/serve_radar.py` → `web/radar.html`

The isomorphic transducer client; uses the `serving.app` control plane (§3) + `/ui`.

### 7.4 Decoupled — `scripts/serve_decoupled.py` → `web/decoupled.html`

The **live global energy‑signature dashboard** for the decoupled engine
(`decoupled_engine/api.py: create_app`). See [09 §7](09-decoupled-engine.md).

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | the live dashboard |
| GET | `/health` | `{device, demo, tension_dim, bodies, coverage, latent_bank}` |
| GET | `/api/coastlines.geojson` | world outline |
| GET | `/api/texture?timestamp=&width=&height=` | raw gamma‑sRGB byte buffer (energy layer), headers `X‑Width/X‑Height/X‑Channels/X‑JD` |
| POST | `/api/point` | `{timestamp,lat,lon}` → `{oklab, rgb, attribution, date}` |
| POST | `/api/similar` | `{timestamp,k}` → nearest historical snapshots by tension‑vector cosine similarity |

`timestamp` accepts a JD number/numeric string or an ISO date/"now". With no
checkpoint the server runs in **demo mode** (random weights, 1950–2050 window) so
the UI is explorable before training finishes.

---

## 8. Dashboards (`web/*.html`)

All self‑contained (no CDN), dark monospace theme, served by a `scripts/serve_*.py`:

| File | Server | What it shows |
|---|---|---|
| `index.html` | `serve.py` | WebGL cosmic‑weather globe (loads `heatmap.json`) |
| `radar.html` | `serve_radar.py` (`/ui`) | isomorphic transducer, dual viewport + inspector |
| `resonance.html` | `serve_resonance.py` | 3‑D topology/resonance viewer |
| `kundali.html` | `serve_kundali.py` | 8‑tier sidereal twin search + 2‑D map |
| `decoupled.html` | `serve_decoupled.py` | live global energy field + click‑to‑inspect |

`web/coastlines.geojson` is the shared world outline; `web/heatmap.json` is a
precomputed real field (2024‑04‑08 eclipse).

---

## 9. Which dependency for which surface

| Surface | Extra |
|---|---|
| `/potential`, `/heatmap`, radar control plane, dashboards | `[serve]` (fastapi, uvicorn) |
| Tokenized index behind `/inspect`… | `[index]` (pyarrow, duckdb, h3, pandas) |
| gRPC `CosmicWeather` | `[grpc]` (grpcio) |
| HDBSCAN clusters in `/inspect` | `[cluster]` (hdbscan, scikit‑learn) |
| Decoupled dashboard | `[train,serve]` (torch + fastapi) |
