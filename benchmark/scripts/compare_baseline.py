#!/usr/bin/env python3
"""
Compare benchmark results against a baseline to detect performance regressions
Exit with non-zero status if significant regressions detected
"""

import json
import sys
from typing import Dict, List, Tuple

# Thresholds for regression detection
THRESHOLDS = {
    'qps': 0.10,        # 10% slower = regression
    'recall': 0.02,     # 2% lower recall = regression  
    'memory': 0.20,     # 20% more memory = warning
    'build_time': 0.50, # 50% slower build = warning
}

class Color:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    BOLD = '\033[1m'
    NC = '\033[0m'

def load_results(filename: str) -> List[Dict]:
    """Load benchmark results from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def find_matching_result(results: List[Dict], library: str, index_type: str) -> Dict:
    """Find result matching library and index type"""
    for result in results:
        if result['Library'] == library and result['IndexType'] == index_type:
            return result
    return None

def compare_metric(current: float, baseline: float, metric_name: str, higher_is_better: bool = True) -> Tuple[str, float, bool]:
    """
    Compare a metric between current and baseline
    Returns (status, percentage_change, is_regression)
    """
    if baseline == 0:
        return "N/A", 0.0, False
    
    change = (current - baseline) / baseline
    
    # Determine if this is a regression
    threshold = THRESHOLDS.get(metric_name, 0.10)
    
    if higher_is_better:
        is_regression = change < -threshold
        is_warning = change < -threshold/2
    else:
        is_regression = change > threshold
        is_warning = change > threshold/2
    
    # Status string with color
    if is_regression:
        status = f"{Color.RED}REGRESSION{Color.NC}"
    elif is_warning:
        status = f"{Color.YELLOW}WARNING{Color.NC}"
    elif abs(change) < 0.01:
        status = f"{Color.GREEN}STABLE{Color.NC}"
    else:
        if (higher_is_better and change > 0) or (not higher_is_better and change < 0):
            status = f"{Color.GREEN}IMPROVED{Color.NC}"
        else:
            status = f"{Color.YELLOW}DEGRADED{Color.NC}"
    
    return status, change * 100, is_regression

def get_metric_value(result: Dict, metric: str) -> float:
    """Extract metric value from result, handling nested structures"""
    if isinstance(result.get(metric), dict):
        return result[metric].get('mean', result[metric])
    return result.get(metric, 0)

def compare_results(current_results: List[Dict], baseline_results: List[Dict]) -> bool:
    """
    Compare current results against baseline
    Returns True if regressions detected
    """
    print(f"\n{Color.BOLD}{'='*120}{Color.NC}")
    print(f"{Color.BOLD}Performance Comparison vs Baseline{Color.NC}")
    print(f"{Color.BOLD}{'='*120}{Color.NC}\n")
    
    has_regressions = False
    total_comparisons = 0
    improvements = 0
    regressions = 0
    warnings = 0
    
    for current in current_results:
        library = current['Library']
        index_type = current['IndexType']
        
        baseline = find_matching_result(baseline_results, library, index_type)
        if not baseline:
            print(f"{Color.YELLOW}⚠ No baseline found for {library} {index_type}{Color.NC}")
            continue
        
        print(f"\n{Color.BOLD}{library} - {index_type}{Color.NC}")
        print("-" * 120)
        
        # Compare QPS (higher is better)
        current_qps = get_metric_value(current, 'QPS')
        baseline_qps = get_metric_value(baseline, 'QPS')
        qps_status, qps_change, qps_regression = compare_metric(
            current_qps, baseline_qps, 'qps', higher_is_better=True
        )
        print(f"  QPS:           {current_qps:>10.0f} vs {baseline_qps:>10.0f}  ({qps_change:>+6.1f}%)  {qps_status}")
        
        # Compare Recall (higher is better)
        current_recall = get_metric_value(current, 'Recall')
        baseline_recall = get_metric_value(baseline, 'Recall')
        recall_status, recall_change, recall_regression = compare_metric(
            current_recall, baseline_recall, 'recall', higher_is_better=True
        )
        print(f"  Recall:        {current_recall:>10.4f} vs {baseline_recall:>10.4f}  ({recall_change:>+6.1f}%)  {recall_status}")
        
        # Compare Average Query Time (lower is better)
        current_query = get_metric_value(current, 'AvgQueryMs')
        baseline_query = get_metric_value(baseline, 'AvgQueryMs')
        query_status, query_change, query_regression = compare_metric(
            current_query, baseline_query, 'query_time', higher_is_better=False
        )
        print(f"  Avg Query:     {current_query:>10.4f} vs {baseline_query:>10.4f} ms ({query_change:>+6.1f}%)  {query_status}")
        
        # Compare P95 Latency (lower is better)
        current_p95 = get_metric_value(current, 'P95QueryMs')
        baseline_p95 = get_metric_value(baseline, 'P95QueryMs')
        p95_status, p95_change, p95_regression = compare_metric(
            current_p95, baseline_p95, 'p95', higher_is_better=False
        )
        print(f"  P95 Latency:   {current_p95:>10.4f} vs {baseline_p95:>10.4f} ms ({p95_change:>+6.1f}%)  {p95_status}")
        
        # Compare Memory (lower is better, but more tolerant)
        current_mem = get_metric_value(current, 'MemoryMB')
        baseline_mem = get_metric_value(baseline, 'MemoryMB')
        mem_status, mem_change, mem_regression = compare_metric(
            current_mem, baseline_mem, 'memory', higher_is_better=False
        )
        print(f"  Memory:        {current_mem:>10.2f} vs {baseline_mem:>10.2f} MB ({mem_change:>+6.1f}%)  {mem_status}")
        
        # Compare Build Time (lower is better, but more tolerant)
        current_build = get_metric_value(current, 'BuildTimeMs')
        baseline_build = get_metric_value(baseline, 'BuildTimeMs')
        build_status, build_change, build_regression = compare_metric(
            current_build, baseline_build, 'build_time', higher_is_better=False
        )
        print(f"  Build Time:    {current_build:>10.0f} vs {baseline_build:>10.0f} ms ({build_change:>+6.1f}%)  {build_status}")
        
        # Track regressions
        total_comparisons += 1
        if qps_regression or recall_regression or query_regression or p95_regression:
            regressions += 1
            has_regressions = True
        elif mem_regression or build_regression:
            warnings += 1
        elif qps_change > 5 or recall_change > 1:  # Significant improvements
            improvements += 1
    
    # Summary
    print(f"\n{Color.BOLD}{'='*120}{Color.NC}")
    print(f"{Color.BOLD}Summary{Color.NC}")
    print(f"{Color.BOLD}{'='*120}{Color.NC}\n")
    
    print(f"Total Comparisons: {total_comparisons}")
    
    if improvements > 0:
        print(f"{Color.GREEN}✓ Improvements:    {improvements}{Color.NC}")
    
    if warnings > 0:
        print(f"{Color.YELLOW}⚠ Warnings:        {warnings}{Color.NC}")
    
    if regressions > 0:
        print(f"{Color.RED}✗ Regressions:     {regressions}{Color.NC}")
    else:
        print(f"{Color.GREEN}✓ No critical regressions detected{Color.NC}")
    
    print()
    
    # Detailed regression info if any
    if has_regressions:
        print(f"\n{Color.RED}{Color.BOLD}CRITICAL REGRESSIONS DETECTED!{Color.NC}\n")
        print("The following indices show significant performance degradation:")
        
        for current in current_results:
            library = current['Library']
            index_type = current['IndexType']
            baseline = find_matching_result(baseline_results, library, index_type)
            
            if baseline:
                issues = []
                
                # Check each metric
                current_qps = get_metric_value(current, 'QPS')
                baseline_qps = get_metric_value(baseline, 'QPS')
                if baseline_qps > 0 and (current_qps - baseline_qps) / baseline_qps < -THRESHOLDS['qps']:
                    issues.append(f"QPS: {current_qps:.0f} vs {baseline_qps:.0f} ({((current_qps - baseline_qps) / baseline_qps * 100):+.1f}%)")
                
                current_recall = get_metric_value(current, 'Recall')
                baseline_recall = get_metric_value(baseline, 'Recall')
                if baseline_recall > 0 and (current_recall - baseline_recall) / baseline_recall < -THRESHOLDS['recall']:
                    issues.append(f"Recall: {current_recall:.4f} vs {baseline_recall:.4f} ({((current_recall - baseline_recall) / baseline_recall * 100):+.1f}%)")
                
                if issues:
                    print(f"\n{Color.RED}• {library} {index_type}:{Color.NC}")
                    for issue in issues:
                        print(f"  - {issue}")
        
        print(f"\n{Color.YELLOW}Recommendation: Investigate these regressions before merging.{Color.NC}\n")
    
    return has_regressions

def generate_comparison_report(current_results: List[Dict], baseline_results: List[Dict], output_file: str):
    """Generate detailed markdown comparison report"""
    
    lines = []
    lines.append("# Benchmark Comparison Report\n\n")
    lines.append("## Summary\n\n")
    lines.append("| Library | Index | QPS Change | Recall Change | Memory Change | Status |\n")
    lines.append("|---------|-------|------------|---------------|---------------|--------|\n")
    
    for current in current_results:
        library = current['Library']
        index_type = current['IndexType']
        baseline = find_matching_result(baseline_results, library, index_type)
        
        if baseline:
            current_qps = get_metric_value(current, 'QPS')
            baseline_qps = get_metric_value(baseline, 'QPS')
            qps_change = ((current_qps - baseline_qps) / baseline_qps * 100) if baseline_qps > 0 else 0
            
            current_recall = get_metric_value(current, 'Recall')
            baseline_recall = get_metric_value(baseline, 'Recall')
            recall_change = ((current_recall - baseline_recall) / baseline_recall * 100) if baseline_recall > 0 else 0
            
            current_mem = get_metric_value(current, 'MemoryMB')
            baseline_mem = get_metric_value(baseline, 'MemoryMB')
            mem_change = ((current_mem - baseline_mem) / baseline_mem * 100) if baseline_mem > 0 else 0
            
            # Determine status
            if qps_change < -THRESHOLDS['qps'] * 100 or recall_change < -THRESHOLDS['recall'] * 100:
                status = "❌ REGRESSION"
            elif qps_change < -THRESHOLDS['qps'] * 50 or mem_change > THRESHOLDS['memory'] * 100:
                status = "⚠️ WARNING"
            elif qps_change > 5 or recall_change > 1:
                status = "✅ IMPROVED"
            else:
                status = "✓ STABLE"
            
            lines.append(f"| {library} | {index_type} | {qps_change:+.1f}% | {recall_change:+.1f}% | {mem_change:+.1f}% | {status} |\n")
    
    lines.append("\n## Detailed Metrics\n\n")
    
    for current in current_results:
        library = current['Library']
        index_type = current['IndexType']
        baseline = find_matching_result(baseline_results, library, index_type)
        
        if baseline:
            lines.append(f"### {library} - {index_type}\n\n")
            
            lines.append("| Metric | Current | Baseline | Change |\n")
            lines.append("|--------|---------|----------|--------|\n")
            
            # QPS
            current_qps = get_metric_value(current, 'QPS')
            baseline_qps = get_metric_value(baseline, 'QPS')
            qps_change = ((current_qps - baseline_qps) / baseline_qps * 100) if baseline_qps > 0 else 0
            lines.append(f"| QPS | {current_qps:.0f} | {baseline_qps:.0f} | {qps_change:+.1f}% |\n")
            
            # Recall
            current_recall = get_metric_value(current, 'Recall')
            baseline_recall = get_metric_value(baseline, 'Recall')
            recall_change = ((current_recall - baseline_recall) / baseline_recall * 100) if baseline_recall > 0 else 0
            lines.append(f"| Recall | {current_recall:.4f} | {baseline_recall:.4f} | {recall_change:+.1f}% |\n")
            
            # Latency
            current_query = get_metric_value(current, 'AvgQueryMs')
            baseline_query = get_metric_value(baseline, 'AvgQueryMs')
            query_change = ((current_query - baseline_query) / baseline_query * 100) if baseline_query > 0 else 0
            lines.append(f"| Avg Query (ms) | {current_query:.4f} | {baseline_query:.4f} | {query_change:+.1f}% |\n")
            
            # Memory
            current_mem = get_metric_value(current, 'MemoryMB')
            baseline_mem = get_metric_value(baseline, 'MemoryMB')
            mem_change = ((current_mem - baseline_mem) / baseline_mem * 100) if baseline_mem > 0 else 0
            lines.append(f"| Memory (MB) | {current_mem:.2f} | {baseline_mem:.2f} | {mem_change:+.1f}% |\n")
            
            lines.append("\n")
    
    with open(output_file, 'w') as f:
        f.writelines(lines)
    
    print(f"\nDetailed comparison report saved to: {output_file}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 compare_baseline.py <current_results.json> <baseline_results.json>")
        print("\nCompare current benchmark results against a baseline to detect regressions.")
        print("Exits with non-zero status if critical regressions are detected.")
        sys.exit(1)
    
    current_file = sys.argv[1]
    baseline_file = sys.argv[2]
    
    try:
        current_results = load_results(current_file)
        baseline_results = load_results(baseline_file)
    except Exception as e:
        print(f"{Color.RED}Error loading results: {e}{Color.NC}")
        sys.exit(1)
    
    # Compare results
    has_regressions = compare_results(current_results, baseline_results)
    
    # Generate detailed report
    report_file = "comparison_report.md"
    generate_comparison_report(current_results, baseline_results, report_file)
    
    # Exit with appropriate status code
    if has_regressions:
        sys.exit(1)  # Fail CI/CD pipeline
    else:
        sys.exit(0)  # Success

if __name__ == "__main__":
    main()