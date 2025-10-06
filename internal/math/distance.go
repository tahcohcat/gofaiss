package math

import (
	"math"
)

// L2Distance computes Euclidean distance between two vectors
func L2Distance(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

// L2DistanceSquared computes squared Euclidean distance (faster, no sqrt)
func L2DistanceSquared(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

// DotProduct computes dot product of two vectors
func DotProduct(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// Norm computes L2 norm (magnitude) of a vector
func Norm(v []float32) float32 {
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	return float32(math.Sqrt(float64(sum)))
}

// NormSquared computes squared L2 norm
func NormSquared(v []float32) float32 {
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	return sum
}

// CosineDistance computes cosine distance (1 - cosine similarity)
func CosineDistance(a, b []float32) float32 {
	dot := DotProduct(a, b)
	normA := Norm(a)
	normB := Norm(b)

	if normA == 0 || normB == 0 {
		return 1.0
	}

	similarity := dot / (normA * normB)
	if similarity > 1 {
		similarity = 1
	} else if similarity < -1 {
		similarity = -1
	}
	return 1 - similarity
}

// CosineDistanceWithNorms computes cosine distance using precomputed norms
func CosineDistanceWithNorms(a, b []float32, normA, normB float32) float32 {
	dot := DotProduct(a, b)

	if normA == 0 || normB == 0 {
		return 1.0
	}

	similarity := dot / (normA * normB)
	if similarity > 1 {
		similarity = 1
	} else if similarity < -1 {
		similarity = -1
	}
	return 1 - similarity
}

// InnerProduct computes negative inner product (for MIPS)
func InnerProduct(a, b []float32) float32 {
	return -DotProduct(a, b)
}

// BatchL2Distance computes L2 distances for multiple queries
func BatchL2Distance(queries [][]float32, target []float32) []float32 {
	results := make([]float32, len(queries))
	for i, query := range queries {
		results[i] = L2Distance(query, target)
	}
	return results
}

// BatchDotProduct computes dot products for multiple queries
func BatchDotProduct(queries [][]float32, target []float32) []float32 {
	results := make([]float32, len(queries))
	for i, query := range queries {
		results[i] = DotProduct(query, target)
	}
	return results
}
