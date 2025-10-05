package hnsw

import (
	"compress/gzip"
	"encoding/gob"
	"os"
)

// HNSW public API placeholder

type HNSWIndex struct {
	// TODO
}

func New(dim int, M int, efConstruction int) *HNSWIndex {
	return &HNSWIndex{}
}

func (idx *HNSWIndex) SaveToFile(filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	gzw := gzip.NewWriter(f)
	defer gzw.Close()

	idx.mu.RLock()
	defer idx.mu.RUnlock()

	encoder := gob.NewEncoder(gzw)
	metadata := []interface{}{
		idx.dim, idx.metric, idx.M, idx.efConstruction,
		idx.efSearch, idx.maxLevel, idx.entryPoint,
		idx.levelMult, idx.nextID,
	}
	for _, m := range metadata {
		if err := encoder.Encode(m); err != nil {
			return err
		}
	}
	if err := encoder.Encode(len(idx.vectors)); err != nil {
		return err
	}
	for _, node := range idx.vectors {
		if err := encoder.Encode(node); err != nil {
			return err
		}
	}
	return nil
}

func (idx *HNSWIndex) LoadFromFile(filename string) error {
	f, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	gzr, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gzr.Close()

	idx.mu.Lock()
	defer idx.mu.Unlock()

	decoder := gob.NewDecoder(gzr)
	if err := decoder.Decode(&idx.dim); err != nil {
		return err
	}
	if err := decoder.Decode(&idx.metric); err != nil {
		return err
	}
	if err := decoder.Decode(&idx.M); err != nil {
		return err
	}
	if err := decoder.Decode(&idx.efConstruction); err != nil {
		return err
	}
	if err := decoder.Decode(&idx.efSearch); err != nil {
		return err
	}
	if err := decoder.Decode(&idx.maxLevel); err != nil {
		return err
	}
	if err := decoder.Decode(&idx.entryPoint); err != nil {
		return err
	}
	if err := decoder.Decode(&idx.levelMult); err != nil {
		return err
	}
	if err := decoder.Decode(&idx.nextID); err != nil {
		return err
	}
	var count int
	if err := decoder.Decode(&count); err != nil {
		return err
	}
	idx.vectors = make(map[int64]*HNSWNode, count)
	for i := 0; i < count; i++ {
		var node HNSWNode
		if err := decoder.Decode(&node); err != nil {
			return err
		}
		idx.vectors[node.ID] = &node
	}
	return nil
}
