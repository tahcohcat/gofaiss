package math

import (
	"math"
)

// SIMD-optimized operations
// This implementation provides fallback scalar operations
// TODO: Add platform-specific SIMD optimizations using assembly or compiler intrinsics
// for AVX2, AVX-512, ARM NEON when available

// SimdL2Distance computes L2 distance with SIMD optimizations when available
func SimdL2Distance(a, b []float32) float32 {
	// Fallback to scalar implementation
	// In production, this would detect CPU features and dispatch to SIMD version
	return L2Distance(a, b)
}

// SimdL2DistanceSquared computes squared L2 distance with SIMD optimizations
func SimdL2DistanceSquared(a, b []float32) float32 {
	// Fallback to scalar implementation
	return L2DistanceSquared(a, b)
}

// SimdDotProduct computes dot product with SIMD optimizations
func SimdDotProduct(a, b []float32) float32 {
	// Fallback to scalar implementation
	return DotProduct(a, b)
}

// SimdCosineDistance computes cosine distance with SIMD optimizations
func SimdCosineDistance(a, b []float32) float32 {
	// Fallback to scalar implementation
	return CosineDistance(a, b)
}

// BatchL2DistanceSimd computes L2 distances for multiple queries with SIMD
func BatchL2DistanceSimd(queries [][]float32, target []float32) []float32 {
	results := make([]float32, len(queries))
	
	// Process queries in parallel-friendly blocks
	// This could be optimized with goroutines for very large batches
	for i, query := range queries {
		results[i] = SimdL2Distance(query, target)
	}
	
	return results
}

// BatchDotProductSimd computes dot products for multiple queries with SIMD
func BatchDotProductSimd(queries [][]float32, target []float32) []float32 {
	results := make([]float32, len(queries))
	
	for i, query := range queries {
		results[i] = SimdDotProduct(query, target)
	}
	
	return results
}

// VectorAdd adds two vectors element-wise with SIMD
func VectorAdd(a, b, result []float32) {
	for i := range a {
		result[i] = a[i] + b[i]
	}
}

// VectorSubtract subtracts two vectors element-wise with SIMD
func VectorSubtract(a, b, result []float32) {
	for i := range a {
		result[i] = a[i] - b[i]
	}
}

// VectorScale multiplies vector by scalar with SIMD
func VectorScale(v []float32, scalar float32, result []float32) {
	for i := range v {
		result[i] = v[i] * scalar
	}
}

// VectorNormalize normalizes vector in-place with SIMD
func VectorNormalize(v []float32) {
	norm := Norm(v)
	if norm == 0 {
		return
	}
	scale := 1.0 / norm
	for i := range v {
		v[i] *= scale
	}
}

// BatchNorm computes norms for multiple vectors
func BatchNorm(vectors [][]float32) []float32 {
	results := make([]float32, len(vectors))
	for i, v := range vectors {
		results[i] = Norm(v)
	}
	return results
}

// MatrixVectorMultiply performs matrix-vector multiplication
// Useful for PQ distance table computation
func MatrixVectorMultiply(matrix [][]float32, vector []float32) []float32 {
	if len(matrix) == 0 {
		return nil
	}
	
	result := make([]float32, len(matrix))
	for i, row := range matrix {
		result[i] = DotProduct(row, vector)
	}
	return result
}

// PairwiseL2Distance computes L2 distances between all pairs of vectors
// Returns a distance matrix where result[i][j] = L2(vectors[i], vectors[j])
func PairwiseL2Distance(vectors [][]float32) [][]float32 {
	n := len(vectors)
	result := make([][]float32, n)
	for i := range result {
		result[i] = make([]float32, n)
	}
	
	for i := 0; i < n; i++ {
		result[i][i] = 0
		for j := i + 1; j < n; j++ {
			dist := L2Distance(vectors[i], vectors[j])
			result[i][j] = dist
			result[j][i] = dist
		}
	}
	
	return result
}

// OptimalBlockSize returns optimal block size for cache efficiency
// Useful for blocking matrix operations
func OptimalBlockSize() int {
	// L1 cache is typically 32-64KB
	// For float32, this is 8K-16K elements
	// Conservative estimate for good cache locality
	return 64
}

// HasSIMDSupport checks if SIMD is available (placeholder)
func HasSIMDSupport() bool {
	// TODO: Detect CPU features (AVX2, AVX-512, NEON)
	return false
}

// GetSIMDInfo returns information about available SIMD extensions
func GetSIMDInfo() map[string]bool {
	// TODO: Implement CPU feature detection
	return map[string]bool{
		"avx":     false,
		"avx2":    false,
		"avx512":  false,
		"neon":    false,
		"sse":     false,
		"sse2":    false,
		"sse3":    false,
		"sse4.1":  false,
		"sse4.2":  false,
	}
}

// AlignedAlloc allocates aligned memory for SIMD operations
// For now, just use regular allocation
func AlignedAlloc(size int) []float32 {
	// TODO: Implement aligned memory allocation for better SIMD performance
	// This would use platform-specific APIs (posix_memalign, _aligned_malloc, etc.)
	return make([]float32, size)
}

// Distance computation optimizations for specific scenarios

// L2DistanceSymmetric computes L2 distance assuming vectors are pre-centered
func L2DistanceSymmetric(a, b []float32, normA, normB float32) float32 {
	// Using the identity: ||a-b||² = ||a||² + ||b||² - 2⟨a,b⟩
	dot := DotProduct(a, b)
	distSq := normA*normA + normB*normB - 2*dot
	if distSq < 0 {
		distSq = 0 // Handle numerical errors
	}
	return float32(math.Sqrt(float64(distSq)))
}

// FastL2DistanceSquaredWithNorms uses precomputed norms for faster computation
func FastL2DistanceSquaredWithNorms(a, b []float32, normSqA, normSqB float32) float32 {
	dot := DotProduct(a, b)
	distSq := normSqA + normSqB - 2*dot
	if distSq < 0 {
		distSq = 0
	}
	return distSq
}