package storage

import (
	"encoding/gob"
	"io"
)

// GobWriter wraps an io.Writer with Gob encoding
type GobWriter struct {
	w       io.Writer
	encoder *gob.Encoder
}

// NewGobWriter creates a new Gob writer
func NewGobWriter(w io.Writer) *GobWriter {
	return &GobWriter{
		w:       w,
		encoder: gob.NewEncoder(w),
	}
}

// Write implements io.Writer
func (gw *GobWriter) Write(p []byte) (n int, err error) {
	return gw.w.Write(p)
}

// Encode encodes a value using Gob
func (gw *GobWriter) Encode(v any) error {
	return gw.encoder.Encode(v)
}

// GobReader wraps an io.Reader with Gob decoding
type GobReader struct {
	r       io.Reader
	decoder *gob.Decoder
}

// NewGobReader creates a new Gob reader
func NewGobReader(r io.Reader) *GobReader {
	return &GobReader{
		r:       r,
		decoder: gob.NewDecoder(r),
	}
}

// Read implements io.Reader
func (gr *GobReader) Read(p []byte) (n int, err error) {
	return gr.r.Read(p)
}

// Decode decodes a value using Gob
func (gr *GobReader) Decode(v any) error {
	return gr.decoder.Decode(v)
}
