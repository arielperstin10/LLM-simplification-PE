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

## Setup

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   - Create a `.env` file in the root directory
   - Set `DATABASE_URL` (see Supabase setup below)
   - Set `OPENAI_API_KEY` or your LLM provider API key
   
   **For Supabase:**
   - Get your connection string from Supabase Dashboard → Settings → Database
   - Use either the direct connection or connection pooler (recommended for production)

4. **Run database migrations:**
   ```bash
   alembic upgrade head
   ```

5. **Run the application:**
   ```bash
   uvicorn app.main:app --reload
   ```

   The API will be available at `http://localhost:8000`
   API documentation: `http://localhost:8000/docs`

## API Endpoints

- `GET /api/v1/health` - Health check endpoint
- `POST /api/v1/simplify/` - Simplify text (to be implemented)

## Development

This backend is designed to work with a separate frontend project. CORS is configured to allow requests from common frontend development ports (3000, 5173).

## Docker

To run with Docker:
```bash
docker-compose up
```
