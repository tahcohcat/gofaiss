package math

import (
	"math"
	"testing"
)

func TestL2Distance(t *testing.T) {
	tests := []struct {
		name     string
		a, b     []float32
		expected float32
		epsilon  float32
	}{
		{
			name:     "identical vectors",
			a:        []float32{1, 2, 3},
			b:        []float32{1, 2, 3},
			expected: 0,
			epsilon:  1e-6,
		},
		{
			name:     "unit distance",
			a:        []float32{0, 0, 0},
			b:        []float32{1, 0, 0},
			expected: 1,
			epsilon:  1e-6,
		},
		{
			name:     "3-4-5 triangle",
			a:        []float32{0, 0},
			b:        []float32{3, 4},
			expected: 5,
			epsilon:  1e-6,
		},
		{
			name:     "negative values",
			a:        []float32{-1, -2, -3},
			b:        []float32{1, 2, 3},
			expected: float32(math.Sqrt(56)), // sqrt((2)^2 + (4)^2 + (6)^2)
			epsilon:  1e-5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := L2Distance(tt.a, tt.b)
			if math.Abs(float64(result-tt.expected)) > float64(tt.epsilon) {
				t.Errorf("L2Distance() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestL2DistanceSquared(t *testing.T) {
	tests := []struct {
		name     string
		a, b     []float32
		expected float32
	}{
		{
			name:     "identical vectors",
			a:        []float32{1, 2, 3},
			b:        []float32{1, 2, 3},
			expected: 0,
		},
		{
			name:     "unit distance squared",
			a:        []float32{0, 0, 0},
			b:        []float32{1, 0, 0},
			expected: 1,
		},
		{
			name:     "distance squared",
			a:        []float32{0, 0},
			b:        []float32{3, 4},
			expected: 25,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := L2DistanceSquared(tt.a, tt.b)
			if result != tt.expected {
				t.Errorf("L2DistanceSquared() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestDotProduct(t *testing.T) {
	tests := []struct {
		name     string
		a, b     []float32
		expected float32
	}{
		{
			name:     "orthogonal vectors",
			a:        []float32{1, 0, 0},
			b:        []float32{0, 1, 0},
			expected: 0,
		},
		{
			name:     "parallel vectors",
			a:        []float32{1, 2, 3},
			b:        []float32{1, 2, 3},
			expected: 14, // 1 + 4 + 9
		},
		{
			name:     "negative dot product",
			a:        []float32{1, 0},
			b:        []float32{-1, 0},
			expected: -1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := DotProduct(tt.a, tt.b)
			if result != tt.expected {
				t.Errorf("DotProduct() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestNorm(t *testing.T) {
	tests := []struct {
		name     string
		v        []float32
		expected float32
		epsilon  float32
	}{
		{
			name:     "unit vector",
			v:        []float32{1, 0, 0},
			expected: 1,
			epsilon:  1e-6,
		},
		{
			name:     "zero vector",
			v:        []float32{0, 0, 0},
			expected: 0,
			epsilon:  1e-6,
		},
		{
			name:     "3-4-5 vector",
			v:        []float32{3, 4},
			expected: 5,
			epsilon:  1e-6,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Norm(tt.v)
			if math.Abs(float64(result-tt.expected)) > float64(tt.epsilon) {
				t.Errorf("Norm() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestCosineDistance(t *testing.T) {
	tests := []struct {
		name     string
		a, b     []float32
		expected float32
		epsilon  float32
	}{
		{
			name:     "identical vectors",
			a:        []float32{1, 2, 3},
			b:        []float32{1, 2, 3},
			expected: 0,
			epsilon:  1e-6,
		},
		{
			name:     "orthogonal vectors",
			a:        []float32{1, 0},
			b:        []float32{0, 1},
			expected: 1,
			epsilon:  1e-6,
		},
		{
			name:     "opposite vectors",
			a:        []float32{1, 0},
			b:        []float32{-1, 0},
			expected: 2,
			epsilon:  1e-6,
		},
		{
			name:     "zero vector a",
			a:        []float32{0, 0},
			b:        []float32{1, 1},
			expected: 1,
			epsilon:  1e-6,
		},
		{
			name:     "zero vector b",
			a:        []float32{1, 1},
			b:        []float32{0, 0},
			expected: 1,
			epsilon:  1e-6,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := CosineDistance(tt.a, tt.b)
			if math.Abs(float64(result-tt.expected)) > float64(tt.epsilon) {
				t.Errorf("CosineDistance() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestBatchL2Distance(t *testing.T) {
	queries := [][]float32{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
	}
	target := []float32{0, 0, 0}

	results := BatchL2Distance(queries, target)

	if len(results) != len(queries) {
		t.Errorf("BatchL2Distance() returned %d results, want %d", len(results), len(queries))
	}

	for i, result := range results {
		expected := float32(1.0)
		if math.Abs(float64(result-expected)) > 1e-6 {
			t.Errorf("BatchL2Distance()[%d] = %v, want %v", i, result, expected)
		}
	}
}

func TestBatchDotProduct(t *testing.T) {
	queries := [][]float32{
		{1, 0, 0},
		{0, 1, 0},
		{1, 1, 0},
	}
	target := []float32{1, 0, 0}

	results := BatchDotProduct(queries, target)

	expected := []float32{1, 0, 1}

	if len(results) != len(queries) {
		t.Errorf("BatchDotProduct() returned %d results, want %d", len(results), len(queries))
	}

	for i, result := range results {
		if result != expected[i] {
			t.Errorf("BatchDotProduct()[%d] = %v, want %v", i, result, expected[i])
		}
	}
}

// Benchmarks
func BenchmarkL2Distance(b *testing.B) {
	a := make([]float32, 128)
	bb := make([]float32, 128)
	for i := range a {
		a[i] = float32(i)
		bb[i] = float32(i + 1)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		L2Distance(a, bb)
	}
}

func BenchmarkDotProduct(b *testing.B) {
	a := make([]float32, 128)
	bb := make([]float32, 128)
	for i := range a {
		a[i] = float32(i)
		bb[i] = float32(i + 1)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		DotProduct(a, bb)
	}
}

func BenchmarkCosineDistance(b *testing.B) {
	a := make([]float32, 128)
	bb := make([]float32, 128)
	for i := range a {
		a[i] = float32(i)
		bb[i] = float32(i + 1)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CosineDistance(a, bb)
	}
}
