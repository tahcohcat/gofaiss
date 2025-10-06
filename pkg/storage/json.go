package storage

import (
	"encoding/json"
	"io"
)

// JSONWriter wraps an io.Writer with JSON encoding
type JSONWriter struct {
	w       io.Writer
	encoder *json.Encoder
}

// NewJSONWriter creates a new JSON writer
func NewJSONWriter(w io.Writer) *JSONWriter {
	encoder := json.NewEncoder(w)
	encoder.SetIndent("", "  ")
	return &JSONWriter{
		w:       w,
		encoder: encoder,
	}
}

// Write implements io.Writer
func (jw *JSONWriter) Write(p []byte) (n int, err error) {
	return jw.w.Write(p)
}

// Encode encodes a value using JSON
func (jw *JSONWriter) Encode(v interface{}) error {
	return jw.encoder.Encode(v)
}

// JSONReader wraps an io.Reader with JSON decoding
type JSONReader struct {
	r       io.Reader
	decoder *json.Decoder
}

// NewJSONReader creates a new JSON reader
func NewJSONReader(r io.Reader) *JSONReader {
	return &JSONReader{
		r:       r,
		decoder: json.NewDecoder(r),
	}
}

// Read implements io.Reader
func (jr *JSONReader) Read(p []byte) (n int, err error) {
	return jr.r.Read(p)
}

// Decode decodes a value using JSON
func (jr *JSONReader) Decode(v interface{}) error {
	return jr.decoder.Decode(v)
}
