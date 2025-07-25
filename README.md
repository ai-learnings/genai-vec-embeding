# Vector Embedding Generation Project

This project demonstrates vector embedding generation using Ollama and MongoDB for storage. It supports both Ollama's local embeddings and VoyageAI's cloud embeddings (commented out in the code).

## Prerequisites

- Python 3.x
- Conda
- MongoDB (running locally or accessible via URI)
- Ollama server running locally

## Setup

1. Clone the repository:
```bash
git clone https://github.com/ai-learnings/genai-vec-embeding.git
cd genai-vec-embeding
```

2. Create and activate Conda environment:
```bash
conda create -n for_gen_ai python=3.13
conda activate for_gen_ai
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy the environment file and configure your settings:
```bash
cp .env.example .env
```
Then edit `.env` with your configuration values.

## Configuration

The following environment variables are required in your `.env` file:

```ini
# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017/?directConnection=true
MONGODB_DATABASE=llm-vec-embeding-db
MONGODB_COLLECTION=embeddings

# Ollama Configuration
OLLAMA_BASE_URL=localhost:11434
OLLAMA_MODEL=mxbai-embed-large

# VoyageAI Configuration (if using)
#VOYAGE_API_KEY=your_voyage_api_key_here
#VOYAGE_MODEL=voyage-3.5
```

### MongoDB Vector Search Index

This project requires a vector search index in MongoDB. Create the following index named `llm-vec-embeding`:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "data_embeded",
      "numDimensions": 1024,
      "similarity": "cosine"
    }
  ]
}
```

You can create this index using MongoDB Compass or the MongoDB shell:

```javascript
db.embeddings.createIndex(
  {
    "data_embeded": {
      "$vectorSearch": {
        "index": "llm-vec-embeding",
        "path": "data_embeded",
        "numDimensions": 1024,
        "similarity": "cosine"
      }
    }
  }
)

## Usage

The project provides functionality to:
1. Generate embeddings from text using Ollama
2. Store embeddings in MongoDB
3. (Optional) Use VoyageAI for cloud-based embeddings

Run the main script:
```bash
python main.py
```

## Features

- Text to vector embedding conversion
- MongoDB integration for storing embeddings
- Support for both local (Ollama) and cloud (VoyageAI) embedding services
- Configurable embedding models
- Type hints for better code maintainability

## Project Structure

```
genai-vec-embeding/
├── .env                 # Environment variables (not in version control)
├── .gitignore          # Git ignore file
├── README.md           # This file
├── requirements.txt    # Project dependencies
└── main.py            # Main application code
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
