# Docker Configuration Guide

This project includes Docker configuration for easy deployment and development.

## Files Overview

- `Docker/Dockerfile` - Main Docker image definition
- `docker-compose.yml` - Production Docker Compose configuration
- `docker-compose.dev.yml` - Development override for hot-reload
- `.dockerignore` - Files to exclude from Docker build context

## Quick Start

### Prerequisites

1. Docker and Docker Compose installed
2. Create a `.env` file in the root directory (see `.env.example` for reference)

### Running with Docker Compose

1. **Start all services (production mode):**
   ```bash
   docker-compose up -d
   ```

2. **Start with development hot-reload:**
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
   ```

3. **View logs:**
   ```bash
   docker-compose logs -f app
   ```

4. **Stop services:**
   ```bash
   docker-compose down
   ```

5. **Stop and remove volumes (clean slate):**
   ```bash
   docker-compose down -v
   ```

## Services

### Application (`app`)
- **Port:** 8000 (configurable via `APP_PORT` env var)
- **Health Check:** `http://localhost:8000/api/v1/health`
- **API Docs:** `http://localhost:8000/docs`

### Database (`db`)
- **Type:** PostgreSQL 15
- **Port:** 5432 (configurable via `POSTGRES_PORT` env var)
- **Default Database:** `llm_simplification`
- **Data Persistence:** Stored in Docker volume `postgres_data`

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
# Application
APP_ENV=production
APP_PORT=8000

# Database
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=llm_simplification
POSTGRES_PORT=5432

# Database URL (auto-generated for docker-compose)
DATABASE_URL=postgresql://postgres:postgres@db:5432/llm_simplification

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# API Keys
OPENAI_API_KEY=your_api_key_here
```

## Building the Docker Image

To build the image manually:

```bash
docker build -f Docker/Dockerfile -t llm-simplification-app .
```

## Running Database Migrations

Migrations run automatically on container startup. To run manually:

```bash
docker-compose exec app alembic upgrade head
```

## Development Workflow

1. **Start services:**
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
   ```

2. **Make code changes** - The app will auto-reload (hot-reload enabled)

3. **Run migrations:**
   ```bash
   docker-compose exec app alembic upgrade head
   ```

4. **Access the API:**
   - API: `http://localhost:8000`
   - Docs: `http://localhost:8000/docs`

## Production Deployment

For production, consider:

1. **Use environment-specific `.env` files**
2. **Set strong database passwords**
3. **Configure proper CORS origins**
4. **Use a reverse proxy (nginx/traefik)**
5. **Enable SSL/TLS**
6. **Set up proper logging and monitoring**

## Troubleshooting

### Database connection issues
- Ensure the database service is healthy: `docker-compose ps`
- Check database logs: `docker-compose logs db`
- Verify `DATABASE_URL` in `.env` matches docker-compose service name

### Port conflicts
- Change `APP_PORT` or `POSTGRES_PORT` in `.env` if ports are already in use

### Migration issues
- Check Alembic configuration
- Ensure database is accessible: `docker-compose exec app alembic current`

### Container won't start
- Check logs: `docker-compose logs app`
- Verify `.env` file exists and has correct values
- Ensure Docker has enough resources allocated

