# GoFAISS Makefile

.PHONY: help build test benchmark lint clean docker-build docker-run docker-dev docker-test docker-benchmark

# Variables
BINARY_NAME=gofaiss
DOCKER_IMAGE=gofaiss
DOCKER_TAG=latest
GO_VERSION=1.25

# Default target
help: ## Show this help message
	@echo "GoFAISS - FAISS-like Vector Search in Go"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

# Local build targets
build: ## Build the CLI binary
	@echo "Building $(BINARY_NAME)..."
	go build -o $(BINARY_NAME) ./cmd/cli

build-all: ## Build all binaries
	@echo "Building all binaries..."
	go build -o $(BINARY_NAME) ./cmd/cli
	go build -o basic-usage ./example/basic_usage.go

test: ## Run tests
	@echo "Running tests..."
	go test -v -race -coverprofile=coverage.txt -covermode=atomic ./...

test-short: ## Run short tests
	@echo "Running short tests..."
	go test -v -short ./...

benchmark: ## Run benchmarks
	@echo "Running benchmarks..."
	go test -bench=. -benchmem -run=^$$ ./...

lint: ## Run linter
	@echo "Running linter..."
	golangci-lint run --timeout=5m

fmt: ## Format code
	@echo "Formatting code..."
	go fmt ./...
	goimports -w .

vet: ## Run go vet
	@echo "Running go vet..."
	go vet ./...

coverage: ## Generate coverage report
	@echo "Generating coverage report..."
	go test -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report generated: coverage.html"

clean: ## Clean build artifacts
	@echo "Cleaning..."
	rm -f $(BINARY_NAME)
	rm -f basic-usage
	rm -f coverage.txt coverage.out coverage.html
	rm -rf benchmark_results/
	find . -name "*.test" -delete
	find . -name "*.out" -delete

# Docker targets
docker-build: ## Build Docker image
	@echo "Building Docker image..."
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-build-dev: ## Build development Docker image
	@echo "Building development Docker image..."
	docker build -t $(DOCKER_IMAGE):dev --target development .

docker-run: ## Run Docker container
	@echo "Running Docker container..."
	docker run --rm -v $(PWD)/data:/app/data -v $(PWD)/indexes:/app/indexes $(DOCKER_IMAGE):$(DOCKER_TAG)

docker-dev: ## Start development environment
	@echo "Starting development environment..."
	docker-compose up -d dev
	docker-compose exec dev bash

docker-test: ## Run tests in Docker
	@echo "Running tests in Docker..."
	docker-compose run --rm test

docker-benchmark: ## Run benchmarks in Docker
	@echo "Running benchmarks in Docker..."
	docker-compose run --rm benchmark

docker-clean: ## Clean Docker images and containers
	@echo "Cleaning Docker..."
	docker-compose down -v
	docker rmi $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_IMAGE):dev || true

# Git targets
git-status: ## Show git status
	@git status

git-diff: ## Show git diff
	@git diff

# Release targets
release-patch: ## Create a patch release (v0.0.x)
	@echo "Creating patch release..."
	@./scripts/release.sh patch

release-minor: ## Create a minor release (v0.x.0)
	@echo "Creating minor release..."
	@./scripts/release.sh minor

release-major: ## Create a major release (vx.0.0)
	@echo "Creating major release..."
	@./scripts/release.sh major

# Documentation
docs-serve: ## Serve documentation locally
	@echo "Serving documentation..."
	cd site && mkdocs serve

docs-build: ## Build documentation
	@echo "Building documentation..."
	cd site && mkdocs build

# Install dependencies
install-tools: ## Install development tools
	@echo "Installing development tools..."
	go install golang.org/x/tools/cmd/goimports@latest
	go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest

# CI targets
ci: lint test build ## Run CI checks locally

ci-full: lint test benchmark build ## Run full CI checks

# Quick targets
quick: fmt vet test-short ## Quick checks before commit

all: clean fmt lint test build ## Run all checks and build
