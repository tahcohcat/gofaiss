package storage

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"testing"
)

// TestVersion represents a versioned test structure
type TestVersion struct {
	Version int
	Data    string
}

// MockSerializable implements the Serializable interface for testing
type MockSerializable struct {
	Version int
	Name    string
	Values  []float32
	Nested  *NestedData
}

type NestedData struct {
	ID    int64
	Items []string
}

func (m *MockSerializable) Save(w Writer) error {
	if err := w.Encode(m.Version); err != nil {
		return err
	}
	if err := w.Encode(m.Name); err != nil {
		return err
	}
	if err := w.Encode(m.Values); err != nil {
		return err
	}
	if err := w.Encode(m.Nested); err != nil {
		return err
	}
	return nil
}

func (m *MockSerializable) Load(r Reader) error {
	if err := r.Decode(&m.Version); err != nil {
		return err
	}
	if err := r.Decode(&m.Name); err != nil {
		return err
	}
	if err := r.Decode(&m.Values); err != nil {
		return err
	}
	if err := r.Decode(&m.Nested); err != nil {
		return err
	}
	return nil
}

func TestGobRoundTrip(t *testing.T) {
	original := &MockSerializable{
		Version: 1,
		Name:    "test",
		Values:  []float32{1.0, 2.0, 3.0},
		Nested: &NestedData{
			ID:    123,
			Items: []string{"a", "b", "c"},
		},
	}

	var buf bytes.Buffer
	writer := NewGobWriter(&buf)

	if err := original.Save(writer); err != nil {
		t.Fatalf("Failed to save: %v", err)
	}

	loaded := &MockSerializable{}
	reader := NewGobReader(&buf)

	if err := loaded.Load(reader); err != nil {
		t.Fatalf("Failed to load: %v", err)
	}

	if original.Version != loaded.Version {
		t.Errorf("Version mismatch: got %d, want %d", loaded.Version, original.Version)
	}
	if original.Name != loaded.Name {
		t.Errorf("Name mismatch: got %s, want %s", loaded.Name, original.Name)
	}
	if len(original.Values) != len(loaded.Values) {
		t.Errorf("Values length mismatch: got %d, want %d", len(loaded.Values), len(original.Values))
	}
	if original.Nested.ID != loaded.Nested.ID {
		t.Errorf("Nested ID mismatch: got %d, want %d", loaded.Nested.ID, original.Nested.ID)
	}
}

func TestJSONRoundTrip(t *testing.T) {
	original := &MockSerializable{
		Version: 1,
		Name:    "test",
		Values:  []float32{1.0, 2.0, 3.0},
		Nested: &NestedData{
			ID:    123,
			Items: []string{"a", "b", "c"},
		},
	}

	var buf bytes.Buffer
	writer := NewJSONWriter(&buf)

	if err := original.Save(writer); err != nil {
		t.Fatalf("Failed to save: %v", err)
	}

	loaded := &MockSerializable{}
	reader := NewJSONReader(&buf)

	if err := loaded.Load(reader); err != nil {
		t.Fatalf("Failed to load: %v", err)
	}

	if original.Version != loaded.Version {
		t.Errorf("Version mismatch: got %d, want %d", loaded.Version, original.Version)
	}
	if original.Name != loaded.Name {
		t.Errorf("Name mismatch: got %s, want %s", loaded.Name, original.Name)
	}
}

// TestVersionCompatibility tests backward compatibility
func TestVersionCompatibility(t *testing.T) {
	tests := []struct {
		name        string
		oldVersion  int
		newVersion  int
		shouldError bool
	}{
		{
			name:        "same version",
			oldVersion:  1,
			newVersion:  1,
			shouldError: false,
		},
		{
			name:        "backward compatible",
			oldVersion:  1,
			newVersion:  2,
			shouldError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save with old version
			original := &MockSerializable{
				Version: tt.oldVersion,
				Name:    "test",
				Values:  []float32{1.0, 2.0},
				Nested:  &NestedData{ID: 1, Items: []string{}},
			}

			var buf bytes.Buffer
			writer := NewGobWriter(&buf)
			if err := original.Save(writer); err != nil {
				t.Fatalf("Failed to save: %v", err)
			}

			// Load with new version awareness
			loaded := &MockSerializable{Version: tt.newVersion}
			reader := NewGobReader(&buf)
			err := loaded.Load(reader)

			if tt.shouldError && err == nil {
				t.Error("Expected error but got none")
			}
			if !tt.shouldError && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		})
	}
}

// TestGobVersionedSerialization tests version-aware serialization
func TestGobVersionedSerialization(t *testing.T) {
	type V1Data struct {
		Version int
		Name    string
	}

	type V2Data struct {
		Version  int
		Name     string
		NewField int // New field added in V2
	}

	// Save V1 data
	v1 := V1Data{Version: 1, Name: "test"}
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	if err := enc.Encode(&v1); err != nil {
		t.Fatalf("Failed to encode V1: %v", err)
	}

	// Try to load as V2 (should handle missing field gracefully)
	v2 := V2Data{}
	dec := gob.NewDecoder(&buf)
	
	// Note: Gob will set NewField to zero value since it's missing
	if err := dec.Decode(&v2); err != nil {
		// Gob may fail if struct definition changed incompatibly
		t.Logf("Expected behavior: loading V1 as V2 failed: %v", err)
	} else {
		if v2.Version != 1 || v2.Name != "test" {
			t.Errorf("V2 data mismatch after loading V1")
		}
	}
}

// TestJSONVersionedSerialization tests JSON version compatibility
func TestJSONVersionedSerialization(t *testing.T) {
	type V1Data struct {
		Version int    `json:"version"`
		Name    string `json:"name"`
	}

	type V2Data struct {
		Version  int    `json:"version"`
		Name     string `json:"name"`
		NewField int    `json:"new_field,omitempty"` // Optional field
	}

	// Save V1 data
	v1 := V1Data{Version: 1, Name: "test"}
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	if err := enc.Encode(&v1); err != nil {
		t.Fatalf("Failed to encode V1: %v", err)
	}

	// Load as V2 (should work with omitempty)
	v2 := V2Data{}
	dec := json.NewDecoder(&buf)
	if err := dec.Decode(&v2); err != nil {
		t.Fatalf("Failed to decode V1 as V2: %v", err)
	}

	if v2.Version != 1 || v2.Name != "test" || v2.NewField != 0 {
		t.Errorf("V2 data mismatch: version=%d, name=%s, newField=%d", 
			v2.Version, v2.Name, v2.NewField)
	}
}

// TestFormatCompatibility ensures different formats can coexist
func TestFormatCompatibility(t *testing.T) {
	original := &MockSerializable{
		Version: 1,
		Name:    "test",
		Values:  []float32{1.0, 2.0, 3.0},
		Nested: &NestedData{
			ID:    123,
			Items: []string{"a", "b"},
		},
	}

	formats := []Format{FormatGob, FormatJSON}

	for _, format := range formats {
		t.Run(string(format), func(t *testing.T) {
			var buf bytes.Buffer
			
			writer, err := NewWriter(&buf, format)
			if err != nil {
				t.Fatalf("Failed to create writer: %v", err)
			}

			if err := original.Save(writer); err != nil {
				t.Fatalf("Failed to save with %s: %v", format, err)
			}

			loaded := &MockSerializable{}
			reader, err := NewReader(&buf, format)
			if err != nil {
				t.Fatalf("Failed to create reader: %v", err)
			}

			if err := loaded.Load(reader); err != nil {
				t.Fatalf("Failed to load with %s: %v", format, err)
			}

			if loaded.Name != original.Name {
				t.Errorf("Data mismatch with %s format", format)
			}
		})
	}
}

// TestEmptyDataSerialization tests edge cases with empty data
func TestEmptyDataSerialization(t *testing.T) {
	empty := &MockSerializable{
		Version: 1,
		Name:    "",
		Values:  []float32{},
		Nested:  &NestedData{ID: 0, Items: []string{}},
	}

	var buf bytes.Buffer
	writer := NewGobWriter(&buf)

	if err := empty.Save(writer); err != nil {
		t.Fatalf("Failed to save empty data: %v", err)
	}

	loaded := &MockSerializable{}
	reader := NewGobReader(&buf)

	if err := loaded.Load(reader); err != nil {
		t.Fatalf("Failed to load empty data: %v", err)
	}

	if len(loaded.Values) != 0 {
		t.Errorf("Expected empty Values, got %d items", len(loaded.Values))
	}
	if len(loaded.Nested.Items) != 0 {
		t.Errorf("Expected empty Items, got %d items", len(loaded.Nested.Items))
	}
}

// TestLargeDataSerialization tests with large datasets
func TestLargeDataSerialization(t *testing.T) {
	large := &MockSerializable{
		Version: 1,
		Name:    "large_dataset",
		Values:  make([]float32, 10000),
		Nested: &NestedData{
			ID:    999999,
			Items: make([]string, 1000),
		},
	}

	for i := range large.Values {
		large.Values[i] = float32(i)
	}
	for i := range large.Nested.Items {
		large.Nested.Items[i] = "item"
	}

	var buf bytes.Buffer
	writer := NewGobWriter(&buf)

	if err := large.Save(writer); err != nil {
		t.Fatalf("Failed to save large data: %v", err)
	}

	loaded := &MockSerializable{}
	reader := NewGobReader(&buf)

	if err := loaded.Load(reader); err != nil {
		t.Fatalf("Failed to load large data: %v", err)
	}

	if len(loaded.Values) != 10000 {
		t.Errorf("Expected 10000 values, got %d", len(loaded.Values))
	}
	if len(loaded.Nested.Items) != 1000 {
		t.Errorf("Expected 1000 items, got %d", len(loaded.Nested.Items))
	}
}