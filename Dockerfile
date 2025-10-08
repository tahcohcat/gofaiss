# Multi-stage Dockerfile for GoFAISS

# Stage 1: Build stage
FROM golang:1.24-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git make ca-certificates

# Set working directory
WORKDIR /build

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download && go mod verify

# Copy source code
COPY . .

# Build the CLI application
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags="-w -s" \
    -o /build/gofaiss-cli \
    ./cmd/cli

# Build example applications (optional)
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags="-w -s" \
    -o /build/basic-usage \
    ./example/basic_usage.go

# Stage 2: Runtime stage (minimal)
FROM alpine:latest AS runtime

# Install runtime dependencies
RUN apk --no-cache add ca-certificates tzdata

# Create non-root user
RUN addgroup -g 1000 gofaiss && \
    adduser -D -u 1000 -G gofaiss gofaiss

# Set working directory
WORKDIR /app

# Copy binaries from builder
COPY --from=builder /build/gofaiss-cli /usr/local/bin/gofaiss
COPY --from=builder /build/basic-usage /usr/local/bin/basic-usage

# Create directories for data
RUN mkdir -p /app/data /app/indexes && \
    chown -R gofaiss:gofaiss /app

# Switch to non-root user
USER gofaiss

# Set volume for persistent data
VOLUME ["/app/data", "/app/indexes"]

# Default command
ENTRYPOINT ["gofaiss"]
CMD ["--help"]

# Stage 3: Development stage (with source code)
FROM golang:1.24-alpine AS development

# Install development tools
RUN apk add --no-cache git make bash curl vim

# Set working directory
WORKDIR /workspace

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Install development tools
RUN go install golang.org/x/tools/gopls@latest && \
    go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest

# Default command for development
CMD ["bash"]
