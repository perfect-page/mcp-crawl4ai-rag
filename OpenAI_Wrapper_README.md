# OpenAI API Wrapper with Robust Retry Logic & Progress Tracking

This module provides a robust wrapper around OpenAI API calls with comprehensive error handling, retry mechanisms, progress tracking, and rate limiting. It's designed to make OpenAI API interactions reliable and user-friendly for batch operations and production environments.

## ✨ Features

- **🔄 Automatic Retry Logic**: Uses `tenacity` for exponential backoff with jitter, retries forever until success
- **📊 Progress Tracking**: Live progress bars using `tqdm` for batch operations
- **📈 Error Metrics**: Tracks and reports error counts by type (HTTP codes, exception types)
- **⚡ Rate Limiting**: Built-in rate limiting and batch processing
- **🎯 Easy Integration**: Drop-in replacement for existing OpenAI calls
- **🔧 Configurable**: Environment variable configuration for behavior customization
- **🚀 Async Support**: Placeholder async versions for future extensibility

## 🚀 Quick Start

### Basic Usage

```python
from src.openai_wrapper import (
    create_embeddings_with_progress,
    call_openai_chat_completion,
    print_error_summary
)

# Create embeddings with progress tracking
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = create_embeddings_with_progress(texts)

# Make chat completion with retry logic
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]
response = call_openai_chat_completion(messages)

# Print error summary
print_error_summary()
```

### Using the Decorator

```python
from src.openai_wrapper import with_openai_retries

@with_openai_retries
def my_custom_openai_function():
    import openai
    return openai.embeddings.create(
        model="text-embedding-3-small",
        input=["Some text"]
    )

# Automatically gets retry logic
result = my_custom_openai_function()
```

## 📋 Configuration

Configure behavior using environment variables:

```bash
# API Configuration
export OPENAI_API_KEY="your-api-key-here"
export MODEL_CHOICE="gpt-4"  # Default model for chat completions

# Wrapper Configuration
export OPENAI_MAX_RETRIES="10"      # Max retries per call (default: 10)
export OPENAI_VERBOSE_LOGS="true"   # Enable verbose logging (default: false)
```

## 🛠️ API Reference

### Core Functions

#### `create_embeddings_with_progress(texts, model, batch_size)`

Creates embeddings for a list of texts with progress tracking.

**Parameters:**
- `texts` (List[str]): List of texts to embed
- `model` (str): OpenAI embedding model (default: "text-embedding-3-small")
- `batch_size` (int): Batch size for processing (default: 100)

**Returns:** List[List[float]] - List of embeddings

**Example:**
```python
texts = [f"Document {i}" for i in range(100)]
embeddings = create_embeddings_with_progress(
    texts=texts,
    model="text-embedding-3-small",
    batch_size=20
)
```

#### `call_openai_chat_completion(messages, model, temperature, max_tokens)`

Creates a chat completion with retry logic.

**Parameters:**
- `messages` (List[Dict[str, str]]): Chat messages
- `model` (str): Model name (default: MODEL_CHOICE env var)
- `temperature` (float): Generation temperature (default: 0.3)
- `max_tokens` (int): Maximum tokens (default: 150)

**Returns:** str - Generated response

**Example:**
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing"}
]
response = call_openai_chat_completion(
    messages=messages,
    temperature=0.7,
    max_tokens=200
)
```

#### `generate_completions_with_progress(prompts, model, temperature, max_tokens, batch_size)`

Generate multiple chat completions with progress tracking.

**Parameters:**
- `prompts` (List[List[Dict[str, str]]]): List of message lists
- `model` (str): Model name
- `temperature` (float): Generation temperature (default: 0.3)
- `max_tokens` (int): Maximum tokens (default: 150)
- `batch_size` (int): Batch size (default: 10)

**Returns:** List[str] - List of generated responses

### Utility Functions

#### `print_error_summary()`
Prints a summary of all errors encountered during API calls.

#### `reset_error_counter()`
Resets the error counter for fresh tracking.

#### `@with_openai_retries`
Decorator that adds retry logic to any function making OpenAI API calls.

## 🎯 Usage Examples

### Example 1: Embedding Website Content

```python
from src.openai_wrapper import (
    create_embeddings_with_progress,
    print_error_summary,
    reset_error_counter
)

# Reset error tracking
reset_error_counter()

# Process crawled website content
website_chunks = [
    "Getting started with our API...",
    "Authentication guide...",
    "Rate limiting information..."
]

# Create embeddings with progress bar
embeddings = create_embeddings_with_progress(
    texts=website_chunks,
    batch_size=10
)

print(f"Created {len(embeddings)} embeddings")
print_error_summary()
```

### Example 2: Batch Content Summarization

```python
from src.openai_wrapper import generate_completions_with_progress

# Prepare prompts for summarization
documents = ["Long document 1...", "Long document 2..."]
prompts = []

for doc in documents:
    prompts.append([
        {"role": "system", "content": "Summarize the following document."},
        {"role": "user", "content": doc}
    ])

# Generate summaries with progress tracking
summaries = generate_completions_with_progress(
    prompts=prompts,
    temperature=0.3,
    max_tokens=100,
    batch_size=5
)
```

### Example 3: Custom Function with Retries

```python
from src.openai_wrapper import with_openai_retries
import openai

@with_openai_retries
def create_contextual_summary(document, chunk):
    """Create a contextual summary with automatic retries."""
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Provide context for this chunk."},
            {"role": "user", "content": f"Document: {document}\nChunk: {chunk}"}
        ]
    )
    return response.choices[0].message.content

# Use with automatic retry logic
summary = create_contextual_summary("Full document...", "Specific chunk...")
```

## 🔄 Retry Logic Details

The wrapper implements robust retry logic using the `tenacity` library:

- **Retry Conditions**: Retries on rate limits, API errors, timeouts, and connection errors
- **Backoff Strategy**: Exponential backoff with jitter (min=1s, max=20s)
- **Retry Policy**: Retries forever until success (configurable via `OPENAI_MAX_RETRIES`)
- **Error Tracking**: Counts and categorizes all retry attempts

### Retriable Exceptions

- `openai.RateLimitError` (HTTP 429)
- `openai.APIError` (HTTP 5xx)
- `openai.InternalServerError`
- `openai.APITimeoutError`
- `openai.APIConnectionError`

## 📊 Progress Tracking

Progress bars are automatically shown for batch operations:

```
Creating embeddings: 100%|██████████| 100/100 [00:45<00:00,  2.22item/s]
Generating completions: 80%|████████  | 8/10 [00:32<00:08,  1.25item/s]
```

## 📈 Error Reporting

At the end of processing, get a comprehensive error summary:

```
=== OpenAI API Error Summary ===
HTTP_429: 15 occurrences
HTTP_500: 3 occurrences
APITimeoutError: 2 occurrences
Total retry attempts: 20
===============================
```

## 🔧 Integration with Existing Code

### Refactoring Old Code

**Before:**
```python
import openai

def old_embedding_function(texts):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]
```

**After:**
```python
from src.openai_wrapper import create_embeddings_with_progress

def new_embedding_function(texts):
    return create_embeddings_with_progress(
        texts=texts,
        model="text-embedding-3-small"
    )
```

### Updating Existing Functions

The wrapper is designed as a drop-in replacement. Simply replace:

- `openai.embeddings.create()` → `call_openai_embeddings()`
- `openai.chat.completions.create()` → `call_openai_chat_completion()`

## 🚀 Demo Script

Run the comprehensive demo to see all features:

```bash
cd src
python demo_openai_wrapper.py
```

The demo includes:
- Embedding creation for 100 content chunks
- Chat completions for different content types
- Batch processing with progress tracking
- Decorator usage examples
- Complete real-world pipeline simulation

## 🔮 Future Enhancements

- **True Async Support**: Implement native async versions using OpenAI's async client
- **Advanced Rate Limiting**: Add more sophisticated rate limiting strategies
- **Caching**: Add optional response caching for duplicate requests
- **Metrics Export**: Export metrics to monitoring systems
- **Custom Retry Strategies**: Allow custom retry policies per function

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**
```python
# Make sure you're importing from the correct path
from src.openai_wrapper import create_embeddings_with_progress
```

**2. API Key Not Set**
```bash
export OPENAI_API_KEY="your-key-here"
```

**3. Model Not Available**
```bash
# Check MODEL_CHOICE environment variable
export MODEL_CHOICE="gpt-3.5-turbo"
```

**4. Rate Limiting**
The wrapper automatically handles rate limits with exponential backoff. If you're still hitting limits, consider:
- Reducing batch sizes
- Adding delays between operations
- Upgrading your OpenAI plan

### Debug Mode

Enable verbose logging for debugging:

```bash
export OPENAI_VERBOSE_LOGS="true"
```

This will show detailed retry attempts and error information.

## 📝 License

This wrapper is part of the mcp-crawl4ai-rag project. See the main project LICENSE for details.

## 🤝 Contributing

When contributing to this wrapper:

1. Maintain backward compatibility
2. Add tests for new features
3. Update documentation
4. Follow the existing error handling patterns
5. Ensure progress tracking works for new batch operations

---

*For more examples and advanced usage, see the `demo_openai_wrapper.py` script.*