# Vector AI

Vector AI is a powerful, easy-to-use library for generating embeddings and using semantic search to identify patterns. It is designed to work cleanly with modern JavaScript and TypeScript codebases.

## Features

- Intuitive API for creating vector embeddings and query matching vector databases for RAG operations
- Support for async operations
- Compatible with both JavaScript and TypeScript

## Installation

You can install Vector AI via npm:

```sh
npm install vector-ai
```

Or with Bun:

```sh
bun add vector-ai
```

## Usage

Here's a quick example of how you can use Vector AI:

```javascript
import { VectorClient } from "vector-ai";

const client = new VectorClient({
  apiKey: "",
  dbUrl: "",
  model: "gpt-4o",
  template: "Your role...",
  temperature: 0.8,
  chunkSize?: 500,
  chunkOverlap?: 100,
});

const question = "What is the capital of France?";

// Create embeddings
const embeddings = await client.createEmbeddings(question);

// Query embeddings
const context = await client.queryEmbeddings(embeddings, "<db function name>"); // e.g., 'match_documents'

// Get answer
const answer = await client.getAnswer(question, context);

```

### Data Ingestion

```javascript
const client = new VectorClient({
  apiKey: "",
  dbUrl: "",
  model: "gpt-3.5-turbo", // gpt-4
  template: "Your role...",
  temperature: 0.8,
  chunkSize?: 500,
  chunkOverlap?: 100,
});
let data = "";
try {
  data = await fs.readFile("test.txt", "utf-8");
} catch (error) {
  console.log(error);
}
try {
  // data and table to insert to
  await client.ingestData(data, "documents");
} catch (error) {
  console.log(error);
}
```

## Contributing

We welcome contributions to Vector AI! Please see our [contributing guide](./CONTRIBUTING.md) for more details.

## License

Vector AI is [MIT licensed](./LICENSE).
