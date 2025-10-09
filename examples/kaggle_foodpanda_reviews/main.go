package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/tahcohcat/gofaiss/pkg/index/hnsw"
	"github.com/tahcohcat/gofaiss/pkg/vector"
)

// Review represents one parsed review
type Review struct {
	Text       string
	Restaurant string
	Cuisine    string
	City       string
}

// loadCSVEmbeddings loads precomputed embeddings from CSV
func loadCSVEmbeddings(path string) ([][]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	records, err := r.ReadAll()
	if err != nil {
		return nil, err
	}

	embs := make([][]float32, len(records))
	for i, rec := range records {
		embs[i] = make([]float32, len(rec))
		for j, v := range rec {
			val, _ := strconv.ParseFloat(v, 32)
			embs[i][j] = float32(val)
		}
	}
	return embs, nil
}

// loadTexts loads review texts from file (one per line)
func loadTexts(path string) ([]Review, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var reviews []Review
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, " | ")
		if len(parts) != 4 {
			fmt.Println("Warning: Skipping malformed line")
			continue
		}

		r := Review{
			Text:       strings.TrimSpace(parts[0]),
			Restaurant: strings.TrimPrefix(strings.TrimSpace(parts[1]), "Restaurant: "),
			Cuisine:    strings.TrimPrefix(strings.TrimSpace(parts[2]), "Cuisine: "),
			City:       strings.TrimPrefix(strings.TrimSpace(parts[3]), "City: "),
		}

		reviews = append(reviews, r)
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return reviews, nil
}

// getEmbeddingViaPython calls a small Python helper script to get embeddings
// This is the simplest solution that guarantees compatibility
func getEmbeddingViaPython(query string) ([]float32, error) {
	// Check if Python is available
	_, err := exec.LookPath("python")
	if err != nil {
		_, err = exec.LookPath("python3")
		if err != nil {
			return nil, fmt.Errorf("python not found. Please install Python and sentence-transformers")
		}
	}

	// Create a temporary Python script if it doesn't exist
	scriptPath := "examples/kaggle_foodpanda_reviews/data/get_embedding.py"
	if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("embedding script not found at %s\nRun: python generate_embedding_script.py", scriptPath)
	}

	// Call Python script
	cmd := exec.Command("python", scriptPath, query)
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("failed to run Python script: %v", err)
	}

	// Parse comma-separated output
	parts := strings.Split(strings.TrimSpace(string(output)), ",")
	embedding := make([]float32, len(parts))
	for i, p := range parts {
		val, err := strconv.ParseFloat(strings.TrimSpace(p), 32)
		if err != nil {
			return nil, fmt.Errorf("failed to parse embedding value: %v", err)
		}
		embedding[i] = float32(val)
	}

	return embedding, nil
}

func main() {
	fmt.Println("🍔🐼 FoodPanda Reviews Semantic Search Demo")
	fmt.Println("==========================================")

	// Load precomputed embeddings
	fmt.Print("Loading embeddings... ")
	embs, err := loadCSVEmbeddings("examples/kaggle_foodpanda_reviews/data/embeddings_1k.csv")
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		fmt.Println("Make sure you've run the Python script to generate embeddings_1k.csv")
		os.Exit(1)
	}
	fmt.Printf("✓ Loaded %d embeddings (dim=%d)\n", len(embs), len(embs[0]))

	// Load corresponding texts
	fmt.Print("Loading reviews... ")
	reviews, err := loadTexts("examples/kaggle_foodpanda_reviews/data/texts_1k.txt")
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		fmt.Println("Make sure you've run the Python script to generate texts_1k.txt")
		os.Exit(1)
	}
	fmt.Printf("✓ Loaded %d reviews\n", len(reviews))

	// Build HNSW index
	fmt.Print("Building HNSW index... ")
	dim := len(embs[0])
	index, err := hnsw.New(dim, "cosine", hnsw.Config{
		M:              16,
		EfConstruction: 200,
		EfSearch:       100,
	})
	if err != nil {
		panic(err)
	}

	// Convert embeddings to vectors
	vectors := make([]vector.Vector, len(embs))
	for i, emb := range embs {
		vectors[i] = vector.Vector{
			ID:   int64(i),
			Data: emb,
		}
	}

	err = index.Add(vectors)
	if err != nil {
		panic(err)
	}
	fmt.Println("✓ Index built")

	// Check for embedding script
	scriptPath := "examples/kaggle_foodpanda_reviews/data/get_embedding.py"
	if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
		fmt.Println("\n  Embedding script not found!")
		fmt.Println("Creating the helper script at:", scriptPath)
		fmt.Println("\nPlease run this command first:")
		fmt.Println("  python examples/kaggle_foodpanda_reviews/data/create_embedding_helper.py")
		fmt.Println("\nOr create get_embedding.py manually with the content shown in the instructions.")
		os.Exit(1)
	}

	// Interactive search loop
	fmt.Println("\n Ready to search! Type your query (or 'quit' to exit)")
	fmt.Println("Example queries:")
	fmt.Println("  - delicious biryani")
	fmt.Println("  - bad service slow delivery")
	fmt.Println("  - best chinese food")
	fmt.Println()

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}
		query := strings.TrimSpace(scanner.Text())

		if query == "" {
			continue
		}
		if query == "quit" || query == "exit" {
			fmt.Println("Goodbye!")
			break
		}

		// Get embedding via Python helper
		fmt.Print("Getting embedding... ")
		qEmb, err := getEmbeddingViaPython(query)
		if err != nil {
			fmt.Printf("\n Error: %v\n\n", err)
			continue
		}
		fmt.Println("✓")

		// Search FAISS
		start := time.Now()
		k := 5
		results, err := index.Search(qEmb, k)
		if err != nil {
			fmt.Printf(" Search error: %v\n\n", err)
			continue
		}
	
		elapsed := float64(time.Since(start).Nanoseconds()) / 1e6

		// Display results
		fmt.Printf("\n Top %d matches [search time:%.9fms]:\n", k, elapsed)
		fmt.Println(strings.Repeat("-", 80))
		for i, result := range results {
			if int(result.ID) < len(reviews) {
				r := reviews[result.ID]
				similarity := 1.0 - result.Distance // Convert distance to similarity
				fmt.Printf("\n%d. [Similarity: %.3f]\n", i+1, similarity)
				fmt.Printf("   Review: %s\n", r.Text)
				fmt.Printf("   Restaurant: %s | Cuisine: %s | City: %s\n",
					r.Restaurant, r.Cuisine, r.City)
			}
		}
		fmt.Println(strings.Repeat("-", 80))
		fmt.Println()
	}
}