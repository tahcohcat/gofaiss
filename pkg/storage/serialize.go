package storage

import (
	"compress/gzip"
	"fmt"
	"io"
	"os"
)

// Format represents serialization format
type Format string

const (
	FormatGob  Format = "gob"
	FormatJSON Format = "json"
)

// Serializable interface for types that can be serialized
type Serializable interface {
	// Save writes the object using a format-agnostic writer
	Save(w Writer) error
	// Load reads the object using a format-agnostic reader
	Load(r Reader) error
}

// SaveToFile saves a Serializable object to a file with specified format and optional compression
func SaveToFile(obj Serializable, filename string, format Format, compress bool) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	var w io.Writer = f
	if compress {
		gzw := gzip.NewWriter(f)
		defer gzw.Close()
		w = gzw
	}

	writer, err := NewWriter(w, format)
	if err != nil {
		return err
	}

	return obj.Save(writer)
}

// LoadFromFile loads a Serializable object from a file with specified format and optional compression
func LoadFromFile(obj Serializable, filename string, format Format, compress bool) error {
	f, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	var r io.Reader = f
	if compress {
		gzr, err := gzip.NewReader(f)
		if err != nil {
			return err
		}
		defer gzr.Close()
		r = gzr
	}

	reader, err := NewReader(r, format)
	if err != nil {
		return err
	}

	return obj.Load(reader)
}

// Writer wraps an io.Writer with format-specific encoding
type Writer interface {
	io.Writer
	Encode(v any) error
}

// Reader wraps an io.Reader with format-specific decoding
type Reader interface {
	io.Reader
	Decode(v any) error
}

// NewWriter creates a format-specific writer
func NewWriter(w io.Writer, format Format) (Writer, error) {
	switch format {
	case FormatGob:
		return NewGobWriter(w), nil
	case FormatJSON:
		return NewJSONWriter(w), nil
	default:
		return nil, fmt.Errorf("unsupported format: %s", format)
	}
}

// NewReader creates a format-specific reader
func NewReader(r io.Reader, format Format) (Reader, error) {
	switch format {
	case FormatGob:
		return NewGobReader(r), nil
	case FormatJSON:
		return NewJSONReader(r), nil
	default:
		return nil, fmt.Errorf("unsupported format: %s", format)
	}
}
