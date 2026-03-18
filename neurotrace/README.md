# NeuroTrace — Parkinson's Intelligence Platform

A real-time, modular web dashboard for Parkinson's Disease monitoring powered by a multi-modal AI fusion engine.

---

## Project Structure

```
neurotrace/
├── index.html                   ← Entry point (loads all modules)
│
├── styles/
│   ├── variables.css            ← Design tokens (colours, fonts, spacing)
│   ├── base.css                 ← CSS reset, body, utilities, tooltips
│   ├── layout.css               ← Sidebar, topbar, main shell, grid system
│   ├── components.css           ← Cards, buttons, alerts, charts, badges, modality...
│   ├── pages.css                ← Page-specific styles (history timeline, realtime)
│   ├── animations.css           ← @keyframes (fadeSlide, pulse, shimmer, spin...)
│   └── responsive.css           ← Breakpoints (1200px, 900px, 768px, 480px)
│
└── js/
    ├── utils/
    │   ├── helpers.js           ← DOM helpers, text/html setters, colour palette
    │   └── charts.js            ← Chart.js defaults, gradient fill, factory (line/bar/radar)
    │
    ├── api/
    │   ├── client.js            ← All fetch() calls to Flask (GET + POST)
    │   └── fallback.js          ← Static fallback data (used when API is unreachable)
    │
    ├── components/
    │   ├── sidebar.js           ← Sidebar HTML + nav events + theme toggle
    │   ├── topbar.js            ← Topbar HTML + patient breadcrumb + refresh
    │   ├── gauge.js             ← SVG severity gauge (template + update)
    │   └── skeletons.js         ← Skeleton loader HTML per page
    │
    ├── pages/
    │   ├── dashboard.js         ← Dashboard: patient card, stats, gauge, charts, alerts
    │   ├── explanation.js       ← Explainability: feature bars, heatmaps, radar chart
    │   ├── recommendations.js   ← Recommendations: filterable priority cards
    │   ├── history.js           ← History: longitudinal charts + intervention timeline
    │   └── realtime.js          ← Real-time: file upload + POST /fusion/realtime_predict
    │
    └── app.js                   ← App state, Router, DOMContentLoaded bootstrap
```

---

## API Endpoints (Flask backend at `http://localhost:5000`)

| Method | Endpoint                         | Used by            |
|--------|----------------------------------|--------------------|
| GET    | `/fusion/dashboard/:patientId`   | Dashboard page     |
| GET    | `/fusion/explanation/:patientId` | Explanation page   |
| GET    | `/fusion/recommendations/:patientId` | Recommendations page |
| GET    | `/fusion/history/:patientId`     | History page       |
| POST   | `/fusion/realtime_predict`       | Real-time page     |

All endpoints gracefully fall back to static sample data when the backend is unreachable.

---

## Running Locally

Simply open `index.html` in a browser. For full API connectivity:

1. Start your Flask backend: `flask run` (port 5000)
2. Open `index.html` via a local server (e.g. VS Code Live Server, `python -m http.server`)

> **Note:** Opening `index.html` directly via `file://` will cause CORS issues when fetching from `localhost:5000`. Use a local HTTP server instead.

---

## Module Responsibilities

| Module              | Responsibility                                                  |
|---------------------|-----------------------------------------------------------------|
| `helpers.js`        | Utility functions: DOM manipulation, text/html setters, palette |
| `charts.js`         | Chart.js defaults, shared configs, gradient fill factory        |
| `client.js`         | All API calls — one function per endpoint                       |
| `fallback.js`       | Rich static data for 4 patients when backend is offline         |
| `sidebar.js`        | Renders sidebar nav, patient selector, theme toggle             |
| `topbar.js`         | Renders top bar, handles refresh + mobile menu                  |
| `gauge.js`          | SVG gauge template injection + live value updates               |
| `skeletons.js`      | Skeleton loader HTML strings per page                           |
| `dashboard.js`      | Patient summary, stat cards, modality grid, progression chart   |
| `explanation.js`    | Feature importance bars, heatmaps, radar attention chart        |
| `recommendations.js`| Filterable recommendation cards with confidence bars            |
| `history.js`        | 3 longitudinal charts + scrollable intervention timeline        |
| `realtime.js`       | Multi-file upload form → POST → fusion result display           |
| `app.js`            | `App` state container, `Router` (navigate/refresh), bootstrap   |
