#!/bin/bash
# Reproducible benchmark runner for GoFAISS
# Ensures consistent environment and multiple runs for statistical validity

set -e

# Configuration
BENCHMARK_BINARY="./benchmark_comparison"
NUM_RUNS=5
OUTPUT_DIR="results/$(date +%Y%m%d_%H%M%S)"
CPUSET="0-7"  # CPU cores to use

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== GoFAISS Reproducible Benchmark Suite ===${NC}\n"

# Check if benchmark binary exists
if [ ! -f "$BENCHMARK_BINARY" ]; then
    echo -e "${YELLOW}Benchmark binary not found. Building...${NC}"
    go build -o benchmark_comparison benchmark_comparison.go
    if [ $? -ne 0 ]; then
        echo -e "${RED}Build failed!${NC}"
        exit 1
    fi
    echo -e "${GREEN}Build successful!${NC}\n"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# System information
echo -e "${YELLOW}System Information:${NC}"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "OS: $(uname -s)"
echo "Kernel: $(uname -r)"
echo "Go Version: $(go version)"
echo "CPU: $(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
echo "CPU Cores: $(nproc)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo ""

# Save system info to file
{
    echo "# System Information"
    echo "Date: $(date)"
    echo "Hostname: $(hostname)"
    echo "OS: $(uname -s) $(uname -r)"
    echo "Go Version: $(go version)"
    echo "CPU: $(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
    echo "CPU Cores: $(nproc)"
    echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
} > "$OUTPUT_DIR/system_info.txt"

# Check CPU governor
CPU_GOVERNOR=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown")
echo -e "${YELLOW}CPU Governor: $CPU_GOVERNOR${NC}"
if [ "$CPU_GOVERNOR" != "performance" ]; then
    echo -e "${YELLOW}Warning: CPU governor is not set to 'performance'${NC}"
    echo -e "Consider running: sudo cpupower frequency-set -g performance"
    echo ""
fi

# Function to run benchmark
run_benchmark() {
    local run_number=$1
    local output_file="$OUTPUT_DIR/run_${run_number}.json"
    
    echo -e "${GREEN}Running benchmark ${run_number}/${NUM_RUNS}...${NC}"
    
    # Use taskset to pin to specific CPUs for consistency
    if command -v taskset &> /dev/null; then
        taskset -c $CPUSET $BENCHMARK_BINARY
    else
        $BENCHMARK_BINARY
    fi
    
    # Move results to output directory
    if [ -f "benchmark_results_10k.json" ]; then
        mv benchmark_results_10k.json "$OUTPUT_DIR/run_${run_number}_10k.json"
    fi
    if [ -f "benchmark_results_100k.json" ]; then
        mv benchmark_results_100k.json "$OUTPUT_DIR/run_${run_number}_100k.json"
    fi
    
    echo -e "${GREEN}Run ${run_number} complete${NC}\n"
}

# Run benchmarks multiple times
echo -e "${YELLOW}Running $NUM_RUNS benchmark iterations...${NC}\n"
for i in $(seq 1 $NUM_RUNS); do
    run_benchmark $i
    
    # Brief pause between runs to let system stabilize
    if [ $i -lt $NUM_RUNS ]; then
        echo "Cooling down for 5 seconds..."
        sleep 5
    fi
done

echo -e "${GREEN}All benchmark runs complete!${NC}\n"

# Aggregate results using Python script (if available)
if command -v python3 &> /dev/null; then
    AGGREGATE_SCRIPT="scripts/aggregate_runs.py"
    if [ -f "$AGGREGATE_SCRIPT" ]; then
        echo -e "${YELLOW}Aggregating results...${NC}"
        python3 $AGGREGATE_SCRIPT "$OUTPUT_DIR"/run_*_10k.json > "$OUTPUT_DIR/aggregated_10k.json"
        python3 $AGGREGATE_SCRIPT "$OUTPUT_DIR"/run_*_100k.json > "$OUTPUT_DIR/aggregated_100k.json"
        echo -e "${GREEN}Aggregation complete${NC}\n"
    fi
    
    # Generate visualizations
    VISUALIZE_SCRIPT="scripts/visualize_benchmark.py"
    if [ -f "$VISUALIZE_SCRIPT" ]; then
        echo -e "${YELLOW}Generating visualizations...${NC}"
        python3 $VISUALIZE_SCRIPT "$OUTPUT_DIR/aggregated_10k.json" || true
        python3 $VISUALIZE_SCRIPT "$OUTPUT_DIR/aggregated_100k.json" || true
        echo -e "${GREEN