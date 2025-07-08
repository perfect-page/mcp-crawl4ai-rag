# OpenAI API Wrapper with Rate Limiting and Retries

A robust Python wrapper for OpenAI API calls that provides automatic rate limiting, infinite retries with exponential backoff, live progress tracking, and comprehensive error reporting.

## Features

✅ **Proactive Rate Limiting** - Prevents exceeding OpenAI's rate limits using `pyrate-limiter`  
✅ **Automatic Retry with Backoff** - Uses `tenacity` for infinite retries with exponential backoff  
✅ **Live Progress Bar** - Real-time progress tracking with `tqdm`  
✅ **Live Error Counter Display** - Dynamic error tracking by HTTP status  
✅ **Reusable Design** - Works with both chat completions and embeddings  
✅ **Comprehensive Logging** - Detailed retry and error event logging  

## Installation

```bash
pip install tenacity tqdm pyrate-limiter openai
```

## Quick Start

### Basic Usage

```python
from openai_wrapper import OpenAIWrapper, call_openai_with_retries

# Single chat completion
response = call_openai_with_retries(
    "What is machine learning?",
    api_type="chat",
    max_tokens=100
)

# Single embedding
embedding = call_openai_with_retries(
    "OpenAI embeddings for semantic search",
    api_type="embedding"
)
```

### Batch Processing with Progress

```python
from openai_wrapper import OpenAIWrapper

# Create wrapper instance
wrapper = OpenAIWrapper(show_progress=True)

# Process multiple texts
texts = ["Text 1", "Text 2", "Text 3"] * 100
embeddings = wrapper.create_embeddings_batch(texts, batch_size=50)

# Print error summary
wrapper.print_error_summary()
```

### Using the Decorator Pattern

```python
from openai_wrapper import with_openai_retries

@with_openai_retries(show_progress=True)
def process_documents(wrapper, documents):
    """Process documents with automatic retry and progress tracking."""
    def summarize(doc):
        return wrapper.chat_completion(
            messages=[
                {"role": "system", "content": "Summarize this document."},
                {"role": "user", "content": doc}
            ],
            max_tokens=100
        )
    
    return wrapper.process_batch_with_progress(
        documents,
        summarize,
        desc="Summarizing documents",
        parallel=True
    )

# Use the decorated function
summaries = process_documents(["Doc 1", "Doc 2", "Doc 3"])
```

## Configuration

### Environment Variables

```bash
# Required
export OPENAI_API_KEY="your-api-key"

# Optional (with defaults)
export OPENAI_RPM_LIMIT=60          # Requests per minute
export OPENAI_TPM_LIMIT=150000      # Tokens per minute
export MODEL_CHOICE="gpt-4"         # Default model for chat
```

### Custom Rate Limits

```python
# Create wrapper with custom rate limit
wrapper = OpenAIWrapper(
    rpm_limit=30,  # 30 requests per minute
    show_progress=True,
    max_parallel_calls=5
)
```

## API Reference

### OpenAIWrapper Class

```python
class OpenAIWrapper:
    def __init__(
        self,
        api_key: Optional[str] = None,
        rpm_limit: Optional[int] = None,
        show_progress: bool = True,
        max_parallel_calls: int = 10
    )
```

**Parameters:**
- `api_key`: OpenAI API key (uses env var if not provided)
- `rpm_limit`: Requests per minute limit
- `show_progress`: Whether to show progress bars
- `max_parallel_calls`: Maximum parallel API calls

### Methods

#### create_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]
Create a single embedding with retry logic.

#### create_embeddings_batch(texts: List[str], model: str = "text-embedding-3-small", batch_size: int = 100) -> List[List[float]]
Create embeddings for multiple texts with progress tracking.

#### chat_completion(messages: List[Dict[str, str]], model: Optional[str] = None, **kwargs) -> str
Create a chat completion with retry logic.

#### process_batch_with_progress(items: List[Any], process_func: Callable, desc: str = "Processing", parallel: bool = True) -> List[Any]
Process a batch of items with progress tracking and parallel execution.

#### print_error_summary()
Print a summary of all errors encountered.

## Error Handling

The wrapper automatically handles these OpenAI exceptions:
- `RateLimitError` (429)
- `APIError`
- `APIConnectionError`
- `APITimeoutError`
- `InternalServerError` (500)

All errors are tracked and can be reviewed:

```python
wrapper = OpenAIWrapper()
# ... make API calls ...
wrapper.print_error_summary()
```

Output:
```
📊 Error Summary:
----------------------------------------
  429: 5 errors
  500: 2 errors
----------------------------------------
Total API calls: 100
Success rate: 93.0%
```

## Live Progress Display

During batch operations, you'll see:

```
Creating embeddings: 45/100 [45%] ██████████░░░░░░░░░░ 
completed: 42, err_429: 3, err_500: 0
```

## Integration with Existing Code

The wrapper is designed to be a drop-in replacement. Update your imports:

```python
# Before
import openai
response = openai.embeddings.create(...)

# After
from openai_wrapper import get_openai_wrapper
wrapper = get_openai_wrapper()
response = wrapper.create_embedding(...)
```

## Advanced Usage

### Parallel Processing

```python
# Process items in parallel
results = wrapper.process_batch_with_progress(
    items=[1, 2, 3, 4, 5],
    process_func=lambda x: wrapper.chat_completion(...),
    desc="Processing items",
    parallel=True  # Enable parallel processing
)
```

### Custom Retry Logic

The wrapper uses exponential backoff with:
- Initial wait: 1 second
- Maximum wait: 60 seconds
- Multiplier: 1 (with jitter)
- Retries: Infinite (until success)

### Logging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Best Practices

1. **Batch Operations**: Use `create_embeddings_batch()` for multiple texts
2. **Parallel Processing**: Enable parallel processing for independent operations
3. **Error Monitoring**: Always check error summary after batch operations
4. **Rate Limits**: Set conservative rate limits for production
5. **Environment Variables**: Use env vars for configuration

## Demo Script

Run the included demo to see all features:

```bash
python demo_openai_wrapper.py
```

The demo includes:
- Single API calls
- Batch embeddings with progress
- Decorator pattern usage
- Error handling demonstration
- Contextual processing

## Troubleshooting

### Common Issues

1. **Rate Limit Errors**: Reduce `rpm_limit` or `batch_size`
2. **Timeout Errors**: Reduce `max_parallel_calls`
3. **Memory Issues**: Process in smaller batches

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## Performance Tips

1. **Optimal Batch Size**: 50-100 texts per embedding batch
2. **Parallel Calls**: 5-10 concurrent calls work well
3. **Rate Limits**: Leave 10-20% buffer below actual limits

## License

This wrapper is part of the mcp-crawl4ai-rag project and follows the same license terms.