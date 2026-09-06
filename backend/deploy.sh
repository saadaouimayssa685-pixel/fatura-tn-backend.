#!/bin/bash

# Invoice OCR API - Docker Build and Run Script

echo "🚀 Building and deploying Invoice OCR API with Docker..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    print_warning ".env file not found. Creating a template..."
    cat > .env << EOF
# Environment variables for Invoice OCR API
GEMINI_API_KEY=your_gemini_api_key_here
UPLOAD_FOLDER=uploads
MAX_FILE_SIZE=10485760
ALLOWED_EXTENSIONS=png,jpg,jpeg,tiff,bmp,webp,pdf
EOF
    print_warning "Please edit .env file with your actual API keys and configuration"
fi

# Check if required model files exist
if [ ! -f "ExtractionModel/best.pt" ]; then
    print_warning "YOLO model file (ExtractionModel/best.pt) not found."
    print_warning "Please ensure your model file is in the ExtractionModel directory."
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p uploads
mkdir -p ExtractionModel
mkdir -p fonts

# Copy font file to fonts directory if it exists
if [ -f "SimSun.ttf" ]; then
    cp SimSun.ttf fonts/
    print_success "Font file copied to fonts directory"
fi

# Stop existing containers
print_status "Stopping existing containers..."
docker-compose down

# Build the Docker image
print_status "Building Docker image..."
if docker-compose build; then
    print_success "Docker image built successfully"
else
    print_error "Failed to build Docker image"
    exit 1
fi

# Start the services
print_status "Starting services..."
if docker-compose up -d; then
    print_success "Services started successfully"
else
    print_error "Failed to start services"
    exit 1
fi

# Wait for the service to be ready
print_status "Waiting for service to be ready..."
sleep 10

# Check if the service is running
if curl -s http://localhost:8002/health > /dev/null; then
    print_success "✅ Invoice OCR API is running successfully!"
    echo ""
    echo "🔗 API URL: http://localhost:8002"
    echo "📚 API Documentation: http://localhost:8002/docs"
    echo "🏥 Health Check: http://localhost:8002/health"
    echo ""
    echo "🐳 Docker Commands:"
    echo "  - View logs: docker-compose logs -f"
    echo "  - Stop services: docker-compose down"
    echo "  - Restart services: docker-compose restart"
    echo "  - View containers: docker-compose ps"
else
    print_error "Service is not responding. Check the logs with: docker-compose logs"
    exit 1
fi
