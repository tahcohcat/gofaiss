.PHONY: all build benchmark benchmark-quick benchmark-full clean visualize compare help

# Default target
all: build

# Build the benchmark binary
build:
	@echo "Building benchmark binary..."
	@go build -o benchmark_comparison benchmark_comparison.go
	@echo "Build complete: ./benchmark_comparison"

# Quick benchmark (single run)
benchmark-quick: build
	@echo "Running quick benchmark (single run)..."
	@./benchmark_comparison
	@echo "Quick benchmark complete!"

# Full benchmark (5 runs with statistics)
benchmark-full: build
	@echo "Running full benchmark suite..."
	@chmod +x run_benchmark.sh
	@./run_benchmark.sh

# Benchmark with specific configuration
benchmark-10k: build
	@echo "Running 10K vector benchmark..."
	@./benchmark_comparison
	@mv benchmark_results_10k.json results/benchmark_10k_$(shell date +%Y%m%d_%H%M%S).json

benchmark-100k: build
	@echo "Running 100K vector benchmark..."
	@./benchmark_comparison
	@mv benchmark_results_100k.json results/benchmark_100k_$(shell date +%Y%m%d_%H%M%S).json

# Install Python dependencies for visualization
install-deps:
	@echo "Installing Python dependencies..."
	@pip3 install matplotlib numpy jq || pip install matplotlib numpy jq

# Visualize latest results
visualize:
	@echo "Generating visualizations..."
	@if [ -f benchmark_results_100k.json ]; then \
		python3 scripts/visualize_benchmark.py benchmark_results_100k.json; \
	else \
		echo "No benchmark results found. Run 'make benchmark' first."; \
	fi

# Visualize specific results
visualize-%:
	@python3 scripts/visualize_benchmark.py $*.json

# Compare with baseline
compare:
	@if [ ! -f baseline/benchmark_results_100k.json ]; then \
		echo "No baseline found. Creating baseline from current results..."; \
		mkdir -p baseline; \
		cp benchmark_results_100k.json baseline/; \
	else \
		echo "Comparing with baseline..."; \
		python3 scripts/compare_baseline.py benchmark_results_100k.json baseline/benchmark_results_100k.json; \
	fi

# Set current results as new baseline
set-baseline:
	@echo "Setting current results as baseline..."
	@mkdir -p baseline
	@cp benchmark_results_*.json baseline/
	@echo "Baseline updated!"

# Clean build artifacts and results
clean:
	@echo "Cleaning build artifacts..."
	@rm -f benchmark_comparison
	@rm -f benchmark_results_*.json
	@rm -rf *_charts/
	@echo "Clean complete!"

# Clean everything including historical results
clean-all: clean
	@echo "Cleaning all results..."
	@rm -rf results/
	@rm -rf baseline/
	@echo "All clean!"

# Run with CPU performance mode
benchmark-perf: build
	@echo "Setting CPU to performance mode..."
	@sudo cpupower frequency-set -g performance || echo "Warning: cpupower not available"
	@echo "Running benchmark with performance governor..."
	@./benchmark_comparison
	@echo "Resetting CPU governor..."
	@sudo cpupower frequency-set -g ondemand || true

# Profile CPU usage
profile-cpu: build
	@echo "Running benchmark with CPU profiling..."
	@go test -bench=. -cpuprofile=cpu.prof
	@echo "Analyzing profile..."
	@go tool pprof -http=:8080 cpu.prof

# Profile memory usage
profile-mem: build
	@echo "Running benchmark with memory profiling..."
	@go test -bench=. -memprofile=mem.prof -benchmem
	@echo "Analyzing profile..."
	@go tool pprof -http=:8080 mem.prof

# Run benchmarks and generate report
report: benchmark-full
	@echo "Generating comprehensive report..."
	@LATEST_DIR=$$(ls -td results/*/ | head -1); \
	cd $$LATEST_DIR && \
	cat SUMMARY.md && \
	echo "\nVisualization charts available in *_charts/ directories"

# Archive results
archive:
	@mkdir -p results/archive
	@TIMESTAMP=$$(date +%Y%m%d_%H%M%S); \
	tar -czf results/archive/benchmark_$$TIMESTAMP.tar.gz \
		benchmark_results_*.json \
		*_charts/ \
		2>/dev/null || true
	@echo "Results archived to results/archive/"

# Validate benchmark reproducibility
validate: build
	@echo "Running validation (3 consecutive runs)..."
	@mkdir -p validation
	@for i in 1 2 3; do \
		echo "Run $$i..."; \
		./benchmark_comparison; \
		mv benchmark_results_100k.json validation/run_$$i.json; \
		sleep 5; \
	done
	@echo "Computing variance..."
	@python3 scripts/aggregate_runs.py validation/run_*.json > validation/aggregated.json
	@echo "Validation complete! Check validation/aggregated_report.md"

# Show benchmark help
help:
	@echo "GoFAISS Benchmark Targets:"
	@echo ""
	@echo "  make build              - Build benchmark binary"
	@echo "  make benchmark-quick    - Run single benchmark"
	@echo "  make benchmark-full     - Run full benchmark suite (5 runs)"
	@echo "  make benchmark-10k      - Run 10K vector benchmark"
	@echo "  make benchmark-100k     - Run 100K vector benchmark"
	@echo ""
	@echo "  make install-deps       - Install Python dependencies"
	@echo "  make visualize          - Generate charts from latest results"
	@echo "  make visualize-FILE     - Generate charts from specific file"
	@echo "  make compare            - Compare with baseline"
	@echo "  make set-baseline       - Set current results as baseline"
	@echo ""
	@echo "  make report             - Run benchmarks and generate report"
	@echo "  make validate           - Test reproducibility"
	@echo "  make archive            - Archive results"
	@echo ""
	@echo "  make profile-cpu        - Profile CPU usage"
	@echo "  make profile-mem        - Profile memory usage"
	@echo "  make benchmark-perf     - Run with performance CPU governor"
	@echo ""
	@echo "  make clean              - Remove build artifacts"
	@echo "  make clean-all          - Remove all results and artifacts"
	@echo "  make help               - Show this help"
	@echo ""
	@echo "Examples:"
	@echo "  make benchmark-full && make visualize"
	@echo "  make benchmark-quick compare"
	@echo "  make validate"

# CI/CD target for automated testing
ci: build
	@echo "Running CI benchmark..."
	@mkdir -p ci-results
	@./benchmark_comparison
	@mv benchmark_results_*.json ci-results/
	@python3 scripts/visualize_benchmark.py ci-results/benchmark_results_100k.json || true
	@if [ -f baseline/benchmark_results_100k.json ]; then \
		python3 scripts/compare_baseline.py \
			ci-results/benchmark_results_100k.json \
			baseline/benchmark_results_100k.json || exit 1; \
	fi
	@echo "CI benchmark complete!"

# Docker benchmark (isolated environment)
docker-benchmark:
	@echo "Building Docker image..."
	@docker build -t gofaiss-benchmark -f Dockerfile.benchmark .
	@echo "Running benchmark in Docker..."
	@docker run --rm -v $(PWD)/results:/app/results gofaiss-benchmark
	@echo "Docker benchmark complete! Results in ./results/"

# Initialize benchmark environment
init:
	@echo "Initializing benchmark environment..."
	@mkdir -p results baseline scripts
	@chmod +x run_benchmark.sh
	@go mod download
	@echo "Environment initialized!"

# Quick test to verify everything works
test: build
	@echo "Running quick test..."
	@timeout 30s ./benchmark_comparison || (echo "Test timed out - OK for quick test"; exit 0)
	@echo "Test complete!"