package testutil

import (
	"bytes"
	"math"
	"testing"

	"github.com/tahcohcat/gofaiss/pkg/storage"
	"github.com/tahcohcat/gofaiss/pkg/vector"
)

// AssertFloat32Equal checks if two float32 values are approximately equal
func AssertFloat32Equal(t *testing.T, got, want, epsilon float32, msg string) {
	t.Helper()
	if math.Abs(float64(got-want)) > float64(epsilon) {
		t.Errorf("%s: got %v, want %v (epsilon %v)", msg, got, want, epsilon)
	}
}

// AssertFloat32SliceEqual checks if two float32 slices are approximately equal
func AssertFloat32SliceEqual(t *testing.T, got, want []float32, epsilon float32, msg string) {
	t.Helper()
	if len(got) != len(want) {
		t.Errorf("%s: length mismatch: got %d, want %d", msg, len(got), len(want))
		return
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > float64(epsilon) {
			t.Errorf("%s: index %d: got %v, want %v", msg, i, got[i], want[i])
		}
	}
}

// AssertNoError checks that error is nil
func AssertNoError(t *testing.T, err error, msg string) {
	t.Helper()
	if err != nil {
		t.Fatalf("%s: unexpected error: %v", msg, err)
	}
}

// AssertError checks that error is not nil
func AssertError(t *testing.T, err error, msg string) {
	t.Helper()
	if err == nil {
		t.Fatalf("%s: expected error but got nil", msg)
	}
}

// AssertEqual checks if two values are equal
func AssertEqual(t *testing.T, got, want interface{}, msg string) {
	t.Helper()
	if got != want {
		t.Errorf("%s: got %v, want %v", msg, got, want)
	}
}

// AssertNotEqual checks if two values are not equal
func AssertNotEqual(t *testing.T, got, notWant interface{}, msg string) {
	t.Helper()
	if got == notWant {
		t.Errorf("%s: got %v, expected different value", msg, got)
	}
}

// AssertTrue checks if condition is true
func AssertTrue(t *testing.T, condition bool, msg string) {
	t.Helper()
	if !condition {
		t.Errorf("%s: expected true, got false", msg)
	}
}

// AssertFalse checks if condition is false
func AssertFalse(t *testing.T, condition bool, msg string) {
	t.Helper()
	if condition {
		t.Errorf("%s: expected false, got true", msg)
	}
}

// GenerateTestVectors creates deterministic test vectors
func GenerateTestVectors(n, dim int) []vector.Vector {
	vectors := make([]vector.Vector, n)
	for i := 0; i < n; i++ {
		data := make([]float32, dim)
		for j := 0; j < dim; j++ {
			// Deterministic pattern for testing
			data[j] = float32(i*dim + j)
		}
		vectors[i] = vector.Vector{
			ID:   int64(i),
			Data: data,
		}
	}
	return vectors
}

// GenerateOrthogonalVectors creates orthogonal unit vectors for testing
func GenerateOrthogonalVectors(dim int) []vector.Vector {
	if dim <= 0 {
		return nil
	}

	vectors := make([]vector.Vector, dim)
	for i := 0; i < dim; i++ {
		data := make([]float32, dim)
		data[i] = 1.0
		vectors[i] = vector.Vector{
			ID:   int64(i),
			Data: data,
		}
	}
	return vectors
}

// TestSerializationRoundTrip tests serialization round-trip for any Serializable
func TestSerializationRoundTrip(t *testing.T, original storage.Serializable, loaded storage.Serializable, format storage.Format) {
	t.Helper()

	var buf bytes.Buffer
	writer, err := storage.NewWriter(&buf, format)
	AssertNoError(t, err, "creating writer")

	err = original.Save(writer)
	AssertNoError(t, err, "saving object")

	reader, err := storage.NewReader(&buf, format)
	AssertNoError(t, err, "creating reader")

	err = loaded.Load(reader)
	AssertNoError(t, err, "loading object")
}

// BenchmarkSetup provides common benchmark setup
type BenchmarkSetup struct {
	Vectors []vector.Vector
	Queries [][]float32
}

// NewBenchmarkSetup creates a benchmark setup with specified parameters
func NewBenchmarkSetup(numVectors, numQueries, dim int, seed int64) *BenchmarkSetup {
	return &BenchmarkSetup{
		Vectors: vector.GenerateRandom(numVectors, dim, seed),
		Queries: generateQueries(numQueries, dim, seed+1),
	}
}

func generateQueries(n, dim int, seed int64) [][]float32 {
	vectors := vector.GenerateRandom(n, dim, seed)
	queries := make([][]float32, n)
	for i, v := range vectors {
		queries[i] = v.Data
	}
	return queries
}

// CompareSearchResults checks if two search result sets are equivalent
func CompareSearchResults(t *testing.T, got, want []vector.SearchResult, msg string) {
	t.Helper()

	if len(got) != len(want) {
		t.Errorf("%s: length mismatch: got %d, want %d", msg, len(got), len(want))
		return
	}

	for i := range got {
		if got[i].ID != want[i].ID {
			t.Errorf("%s: result %d: ID mismatch: got %d, want %d", msg, i, got[i].ID, want[i].ID)
		}
		AssertFloat32Equal(t, got[i].Distance, want[i].Distance, 1e-5,
			msg+" result "+string(rune(i))+" distance")
	}
}

// IsValidFloat32 checks if a float32 is valid (not NaN or Inf)
func IsValidFloat32(v float32) bool {
	return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0)
}

// IsValidFloat32Slice checks if all values in a slice are valid
func IsValidFloat32Slice(v []float32) bool {
	for _, x := range v {
		if !IsValidFloat32(x) {
			return false
		}
	}
	return true
}

// AssertValidSearchResults verifies search results are valid
func AssertValidSearchResults(t *testing.T, results []vector.SearchResult, msg string) {
	t.Helper()

	for i, r := range results {
		if !IsValidFloat32(r.Distance) {
			t.Errorf("%s: result %d has invalid distance: %v", msg, i, r.Distance)
		}
		if r.Distance < 0 {
			t.Errorf("%s: result %d has negative distance: %v", msg, i, r.Distance)
		}
	}

	// Verify results are sorted by distance
	for i := 1; i < len(results); i++ {
		if results[i].Distance < results[i-1].Distance {
			t.Errorf("%s: results not sorted at index %d: %v > %v",
				msg, i, results[i-1].Distance, results[i].Distance)
		}
	}
}

// SkipIfShort skips test if running in short mode
func SkipIfShort(t *testing.T, reason string) {
	t.Helper()
	if testing.Short() {
		t.Skipf("Skipping in short mode: %s", reason)
	}
}

// MeasureTime measures execution time of a function
func MeasureTime(t *testing.T, name string, fn func()) {
	t.Helper()
	start := t.Now()
	fn()
	duration := t.Since(start)
	t.Logf("%s took %v", name, duration)
}

// RepeatTest runs a test multiple times (useful for flaky test detection)
func RepeatTest(t *testing.T, n int, testFunc func(t *testing.T)) {
	t.Helper()
	for i := 0; i < n; i++ {
		t.Run("iteration_"+string(rune(i)), testFunc)
	}
}

// CreateTempIndex creates a temporary index for testing
type TempIndex interface {
	Cleanup()
}

// AssertPanics checks that a function panics
func AssertPanics(t *testing.T, fn func(), msg string) {
	t.Helper()
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("%s: expected panic but didn't panic", msg)
		}
	}()
	fn()
}

// AssertNoPanics checks that a function doesn't panic
func AssertNoPanics(t *testing.T, fn func(), msg string) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("%s: unexpected panic: %v", msg, r)
		}
	}()
	fn()
}

// WithinTolerance checks if a value is within tolerance of target
func WithinTolerance(value, target, tolerance float32) bool {
	return math.Abs(float64(value-target)) <= float64(tolerance)
}

// AssertWithinTolerance asserts value is within tolerance
func AssertWithinTolerance(t *testing.T, value, target, tolerance float32, msg string) {
	t.Helper()
	if !WithinTolerance(value, target, tolerance) {
		t.Errorf("%s: value %v not within tolerance %v of target %v",
			msg, value, tolerance, target)
	}
}

// CreateTestMatrix creates a test matrix of float32 values
func CreateTestMatrix(rows, cols int) [][]float32 {
	matrix := make([][]float32, rows)
	for i := range matrix {
		matrix[i] = make([]float32, cols)
		for j := range matrix[i] {
			matrix[i][j] = float32(i*cols + j)
		}
	}
	return matrix
}

// AssertMatrixEqual compares two matrices with tolerance
func AssertMatrixEqual(t *testing.T, got, want [][]float32, epsilon float32, msg string) {
	t.Helper()

	if len(got) != len(want) {
		t.Errorf("%s: row count mismatch: got %d, want %d", msg, len(got), len(want))
		return
	}

	for i := range got {
		if len(got[i]) != len(want[i]) {
			t.Errorf("%s: row %d col count mismatch: got %d, want %d",
				msg, i, len(got[i]), len(want[i]))
			continue
		}

		for j := range got[i] {
			if math.Abs(float64(got[i][j]-want[i][j])) > float64(epsilon) {
				t.Errorf("%s: [%d][%d]: got %v, want %v",
					msg, i, j, got[i][j], want[i][j])
			}
		}
	}
}

// RandomFloat32InRange generates a random float32 in [min, max]
func RandomFloat32InRange(min, max float32) float32 {
	// Note: For tests, use deterministic generation via vector.GenerateRandom
	// This is a helper for specific test cases
	return min + (max-min)*0.5 // Placeholder, use proper random in actual code
}

// VectorDistance computes L2 distance between two vectors (test helper)
func VectorDistance(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

// FindClosestVector finds the ID of the closest vector in a set
func FindClosestVector(query []float32, vectors []vector.Vector) int64 {
	if len(vectors) == 0 {
		return -1
	}

	minDist := float32(math.Inf(1))
	minID := vectors[0].ID

	for _, v := range vectors {
		dist := VectorDistance(query, v.Data)
		if dist < minDist {
			minDist = dist
			minID = v.ID
		}
	}

	return minID
}

// AssertSearchAccuracy checks if search results match expected closest vectors
func AssertSearchAccuracy(t *testing.T, query []float32, results []vector.SearchResult,
	allVectors []vector.Vector, k int, msg string) {
	t.Helper()

	expectedID := FindClosestVector(query, allVectors)

	found := false
	for i := 0; i < k && i < len(results); i++ {
		if results[i].ID == expectedID {
			found = true
			break
		}
	}

	if !found {
		t.Errorf("%s: expected to find vector %d in top-%d results", msg, expectedID, k)
	}
}
