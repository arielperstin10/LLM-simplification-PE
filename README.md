# LLM Simplification Backend

Backend API for text simplification using Large Language Models (LLMs) and prompt engineering.

## Project Structure

```
app/
├── api/              # API endpoints
│   └── v1/
│       ├── endpoints/
│       │   ├── health.py
│       │   └── simplification.py
│       └── router.py
├── core/             # Core configuration
│   ├── config.py
│   └── logging.py
├── db/               # Database setup
│   ├── base.py
│   └── session.py
├── models/           # SQLAlchemy models
│   ├── request.py
│   └── user.py
├── repositories/     # Data access layer
│   └── request_repo.py
├── services/         # Business logic
│   └── simplifier.py
└── main.py           # FastAPI application entry point
```

## Docker

To run with Docker:

```bash
docker-compose up
```

## Project URLs

### Test Environment

- **URL:** [https://llm-simplification-api-test.onrender.com](https://llm-simplification-api-test.onrender.com)
- **Branch:** `testServer`
- **Health Check:** [https://llm-simplification-api-test.onrender.com/api/v1/health](https://llm-simplification-api-test.onrender.com/api/v1/health)
- **DB Connection Check:** [https://llm-simplification-api-test.onrender.com/api/v1/health/db](https://llm-simplification-api-test.onrender.com/api/v1/health/db)

### Production Environment

- **URL:** [https://llm-simplification-api.onrender.com](https://llm-simplification-api.onrender.com)
- **Branch:** `main`
- **Health Check:** [https://llm-simplification-api.onrender.com/api/v1/health](https://llm-simplification-api.onrender.com/api/v1/health)
- **DB Connection Check:** [https://llm-simplification-api.onrender.com/api/v1/health/db](https://llm-simplification-api.onrender.com/api/v1/health/db)