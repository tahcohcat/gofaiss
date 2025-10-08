# Docker Guide

This guide explains how to use GoFAISS with Docker.

## Quick Start

### Build the Docker Image

```bash
docker build -t gofaiss:latest .
```

### Run a Benchmark

```bash
docker run --rm gofaiss:latest bench --type hnsw --vectors 10000 --dim 128
```

### Run with Persistent Data

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/indexes:/app/indexes \
  gofaiss:latest bench --type hnsw --vectors 10000
```

## Docker Images

The Dockerfile provides three build targets:

### 1. Runtime (Production)

Minimal Alpine-based image (~15MB):

```bash
docker build -t gofaiss:latest --target runtime .
```

**Features:**
- Small image size
- Non-root user
- Only includes compiled binaries
- Best for production deployments

**Usage:**
```bash
docker run --rm gofaiss:latest --help
```

### 2. Development

Full development environment with Go toolchain:

```bash
docker build -t gofaiss:dev --target development .
```

**Features:**
- Full Go toolchain
- Development tools (gopls, golangci-lint)
- Source code included
- Shell access for development

**Usage:**
```bash
docker run -it --rm -v $(pwd):/workspace gofaiss:dev bash
```

### 3. Builder

Intermediate build stage (not typically used directly).

## Docker Compose

Use Docker Compose for easier management:

### Start Services

```bash
# Build all services
docker-compose build

# Run production benchmark
docker-compose up gofaiss

# Start development shell
docker-compose up -d dev
docker-compose exec dev bash

# Run tests
docker-compose run --rm test

# Run benchmarks
docker-compose run --rm benchmark
```

### Stop Services

```bash
docker-compose down
```

### Clean Up

```bash
# Remove containers and volumes
docker-compose down -v

# Remove images
docker rmi gofaiss:latest gofaiss:dev
```

## Common Use Cases

### 1. Run Tests in Docker

```bash
docker-compose run --rm test
```

Or manually:

```bash
docker run --rm -v $(pwd):/workspace gofaiss:dev \
  go test -v ./...
```

### 2. Run Benchmarks

```bash
docker-compose run --rm benchmark
```

Or manually:

```bash
docker run --rm -v $(pwd):/workspace gofaiss:dev \
  go test -bench=. -benchmem ./...
```

### 3. Development Environment

```bash
# Start dev container
docker-compose up -d dev

# Attach to shell
docker-compose exec dev bash

# Inside container:
go test ./...
go build ./cmd/cli
./cli bench --type hnsw --vectors 5000
```

### 4. Build Index from Data

```bash
# Prepare your vectors
mkdir -p data indexes

# Run index builder
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/indexes:/app/indexes \
  gofaiss:latest build \
    --type hnsw \
    --input /app/data/vectors.json \
    --output /app/indexes/my_index.faiss \
    --dim 128
```

### 5. Search with Existing Index

```bash
docker run --rm \
  -v $(pwd)/indexes:/app/indexes \
  gofaiss:latest search \
    --index /app/indexes/my_index.faiss \
    --query /app/data/query.json \
    --k 10
```

## Using Makefile

The Makefile provides convenient Docker commands:

```bash
# Build Docker images
make docker-build
make docker-build-dev

# Run in Docker
make docker-run
make docker-dev
make docker-test
make docker-benchmark

# Clean Docker artifacts
make docker-clean
```

## Environment Variables

You can customize behavior with environment variables:

```bash
docker run --rm \
  -e GOFAISS_LOG_LEVEL=debug \
  -e GOFAISS_THREADS=4 \
  gofaiss:latest bench --type hnsw --vectors 10000
```

## Volume Mounts

### Data Directory

Mount your data directory to `/app/data`:

```bash
-v /path/to/data:/app/data
```

### Indexes Directory

Mount your indexes directory to `/app/indexes`:

```bash
-v /path/to/indexes:/app/indexes
```

### Complete Example

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/indexes:/app/indexes \
  -v $(pwd)/results:/app/results \
  gofaiss:latest bench \
    --type hnsw \
    --vectors 50000 \
    --dim 256 \
    --output /app/results/benchmark.json
```

## Multi-Architecture Support

Build for different architectures:

### AMD64 (x86_64)

```bash
docker build -t gofaiss:amd64 --build-arg GOARCH=amd64 .
```

### ARM64 (Apple Silicon, ARM servers)

```bash
docker build -t gofaiss:arm64 --build-arg GOARCH=arm64 .
```

### Multi-platform with Buildx

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t gofaiss:latest \
  --push .
```

## Performance Optimization

### 1. Layer Caching

The Dockerfile is optimized for layer caching:
- Dependencies are downloaded before copying source
- Changes to code don't invalidate dependency cache

### 2. Multi-stage Build

Only runtime dependencies are in the final image:
- Builder stage: ~800MB
- Runtime stage: ~15MB

### 3. Parallel Builds

Use BuildKit for faster builds:

```bash
DOCKER_BUILDKIT=1 docker build -t gofaiss:latest .
```

### 4. Resource Limits

Limit resources for benchmarks:

```bash
docker run --rm \
  --cpus=4 \
  --memory=8g \
  gofaiss:latest bench --vectors 100000
```

## Debugging

### Access Container Shell

```bash
docker run -it --rm --entrypoint /bin/sh gofaiss:latest
```

### Inspect Image

```bash
docker inspect gofaiss:latest
docker history gofaiss:latest
```

### View Logs

```bash
docker-compose logs gofaiss
docker-compose logs -f dev
```

### Debug Build

Build with debug info:

```bash
docker build -t gofaiss:debug --target development .
docker run -it --rm gofaiss:debug bash
```

## Best Practices

1. **Use Multi-stage Builds**: Keep production images small
2. **Non-root User**: Always run as non-root in production
3. **Layer Caching**: Order Dockerfile commands for optimal caching
4. **Version Tags**: Tag images with versions, not just `latest`
5. **Health Checks**: Add health checks for long-running services
6. **Secrets**: Never include secrets in images
7. **Scanning**: Scan images for vulnerabilities

## CI/CD Integration

### GitHub Actions

```yaml
- name: Build Docker image
  run: docker build -t gofaiss:${{ github.sha }} .

- name: Run tests in Docker
  run: docker run --rm gofaiss:${{ github.sha }} test

- name: Push to registry
  run: |
    docker tag gofaiss:${{ github.sha }} myregistry/gofaiss:latest
    docker push myregistry/gofaiss:latest
```

### GitLab CI

```yaml
docker-build:
  stage: build
  script:
    - docker build -t gofaiss:latest .
    - docker run --rm gofaiss:latest test
```

## Troubleshooting

### Issue: "permission denied"

Make sure you're running as the correct user:

```bash
docker run --rm --user $(id -u):$(id -g) gofaiss:latest
```

### Issue: "no space left on device"

Clean up Docker:

```bash
docker system prune -a
docker volume prune
```

### Issue: Build is slow

Enable BuildKit:

```bash
export DOCKER_BUILDKIT=1
```

### Issue: Container exits immediately

Check logs:

```bash
docker logs <container-id>
```

Run interactively:

```bash
docker run -it --rm gofaiss:latest bash
```

## Additional Resources

- [Dockerfile Reference](https://docs.docker.com/engine/reference/builder/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Best Practices](https://docs.docker.com/develop/dev-best-practices/)
