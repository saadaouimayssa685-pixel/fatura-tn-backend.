# Invoice OCR API - Makefile

.PHONY: help build up down logs test clean dev prod

# Default target
help:
	@echo "🚀 Invoice OCR API - Docker Commands"
	@echo ""
	@echo "Development:"
	@echo "  make build     - Build Docker image"
	@echo "  make up        - Start services"
	@echo "  make down      - Stop services"
	@echo "  make logs      - View logs"
	@echo "  make test      - Run API tests"
	@echo "  make dev       - Full development setup"
	@echo ""
	@echo "Production:"
	@echo "  make prod      - Production deployment"
	@echo "  make prod-down - Stop production"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean     - Clean up containers and images"
	@echo "  make reset     - Reset everything (careful!)"

# Development commands
build:
	@echo "🔨 Building Docker image..."
	docker-compose build

up:
	@echo "🚀 Starting services..."
	docker-compose up -d

down:
	@echo "🛑 Stopping services..."
	docker-compose down

logs:
	@echo "📋 Viewing logs..."
	docker-compose logs -f

test:
	@echo "🧪 Running API tests..."
	@sleep 5
	@./test_api.sh

dev: build up test
	@echo "✅ Development environment ready!"
	@echo "🌐 API: http://localhost:8002"
	@echo "📚 Docs: http://localhost:8002/docs"

# Production commands
prod:
	@echo "🏭 Starting production deployment..."
	docker-compose -f docker-compose.prod.yml up --build -d

prod-down:
	@echo "🛑 Stopping production..."
	docker-compose -f docker-compose.prod.yml down

# Maintenance commands
clean:
	@echo "🧹 Cleaning up..."
	docker-compose down
	docker system prune -f
	docker volume prune -f

reset: down clean
	@echo "🔄 Resetting everything..."
	docker-compose down -v
	docker system prune -af
	docker volume prune -f

# Utility commands
shell:
	@echo "🐚 Opening shell in container..."
	docker-compose exec invoice-ocr-api bash

restart:
	@echo "🔄 Restarting services..."
	docker-compose restart

status:
	@echo "📊 Container status..."
	docker-compose ps

env:
	@echo "🔧 Checking environment..."
	@if [ ! -f .env ]; then \
		echo "⚠️  .env file not found. Creating template..."; \
		cp .env.example .env 2>/dev/null || echo "GEMINI_API_KEY=your_api_key_here" > .env; \
	fi
	@echo "✅ Environment configured"

# Docker health check
health:
	@echo "🏥 Checking API health..."
	@curl -s http://localhost:8002/health | jq . || echo "❌ API not responding"

# View container stats
stats:
	@echo "📈 Container statistics..."
	docker stats --no-stream
