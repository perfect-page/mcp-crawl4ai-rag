"""
Robust OpenAI API wrapper with retry logic, rate limiting, and progress tracking.

This module provides utilities for making OpenAI API calls with comprehensive
error handling, retry mechanisms, and progress tracking for batch operations.
"""

import os
import logging
import asyncio
from collections import Counter
from typing import List, Dict, Any, Optional, Union, Callable
from functools import wraps

import openai
from tenacity import (
    retry,
    stop_never,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log
)
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)

# Configuration from environment variables
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "10"))
OPENAI_VERBOSE_LOGS = os.getenv("OPENAI_VERBOSE_LOGS", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Error tracking
error_counter = Counter()

# Set OpenAI API key
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Define retriable OpenAI exceptions
RETRIABLE_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIError,
    openai.InternalServerError,
    openai.APITimeoutError,
    openai.APIConnectionError,
)


def track_error(exception: Exception) -> None:
    """Track errors for metrics collection."""
    if hasattr(exception, 'status_code'):
        error_counter[f"HTTP_{exception.status_code}"] += 1
    else:
        error_counter[type(exception).__name__] += 1


def log_retry_attempt(retry_state) -> None:
    """Log retry attempts with details."""
    if OPENAI_VERBOSE_LOGS:
        logger.warning(
            f"OpenAI API call failed (attempt {retry_state.attempt_number}): "
            f"{retry_state.outcome.exception()}"
        )
        track_error(retry_state.outcome.exception())


def log_success(retry_state) -> None:
    """Log successful API calls after retries."""
    if retry_state.attempt_number > 1 and OPENAI_VERBOSE_LOGS:
        logger.info(
            f"OpenAI API call succeeded after {retry_state.attempt_number} attempts"
        )


def create_retry_decorator():
    """Create a retry decorator for OpenAI API calls."""
    return retry(
        retry=retry_if_exception_type(RETRIABLE_EXCEPTIONS),
        wait=wait_exponential(multiplier=1, min=1, max=20, exp_base=2),
        stop=stop_never,  # Retry forever as requested
        before_sleep=before_sleep_log(logger, logging.WARNING) if OPENAI_VERBOSE_LOGS else None,
        after=after_log(logger, logging.INFO) if OPENAI_VERBOSE_LOGS else log_success,
    )


# Apply retry decorator
retry_openai_call = create_retry_decorator()


@retry_openai_call
def call_openai_embeddings(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """
    Create embeddings for multiple texts with retry logic.
    
    Args:
        texts: List of texts to create embeddings for
        model: OpenAI embedding model to use
        
    Returns:
        List of embeddings (each embedding is a list of floats)
        
    Raises:
        OpenAI API exceptions (will be retried automatically)
    """
    try:
        response = openai.embeddings.create(
            model=model,
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        track_error(e)
        raise


@retry_openai_call
def call_openai_chat_completion(
    messages: List[Dict[str, str]], 
    model: str = None,
    temperature: float = 0.3,
    max_tokens: int = 150
) -> str:
    """
    Create a chat completion with retry logic.
    
    Args:
        messages: List of message dictionaries
        model: OpenAI model to use (defaults to MODEL_CHOICE env var)
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        
    Returns:
        Generated text response
        
    Raises:
        OpenAI API exceptions (will be retried automatically)
    """
    if model is None:
        model = os.getenv("MODEL_CHOICE", "gpt-3.5-turbo")
    
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        track_error(e)
        raise


def call_openai_with_retries(
    func: Callable,
    *args,
    progress_desc: str = "Processing",
    **kwargs
) -> Any:
    """
    Generic wrapper for OpenAI API calls with progress tracking.
    
    Args:
        func: OpenAI API function to call
        *args: Positional arguments for the function
        progress_desc: Description for progress bar
        **kwargs: Keyword arguments for the function
        
    Returns:
        Result from the OpenAI API call
    """
    # For single calls, we don't need a progress bar
    return func(*args, **kwargs)


def batch_process_with_progress(
    items: List[Any],
    process_func: Callable,
    batch_size: int = 20,
    progress_desc: str = "Processing items",
    **kwargs
) -> List[Any]:
    """
    Process items in batches with progress tracking and error handling.
    
    Args:
        items: List of items to process
        process_func: Function to process each batch
        batch_size: Size of each batch
        progress_desc: Description for the progress bar
        **kwargs: Additional arguments for process_func
        
    Returns:
        List of results from processing all batches
    """
    results = []
    total_batches = (len(items) + batch_size - 1) // batch_size
    
    with tqdm(total=len(items), desc=progress_desc, unit="item") as pbar:
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            try:
                batch_result = process_func(batch, **kwargs)
                results.extend(batch_result if isinstance(batch_result, list) else [batch_result])
                pbar.update(len(batch))
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}/{total_batches}: {e}")
                # For embedding batch failures, create fallback embeddings
                if "embedding" in progress_desc.lower():
                    fallback_embeddings = [[0.0] * 1536] * len(batch)
                    results.extend(fallback_embeddings)
                    pbar.update(len(batch))
                else:
                    raise
    
    return results


def create_embeddings_with_progress(
    texts: List[str], 
    model: str = "text-embedding-3-small",
    batch_size: int = 100
) -> List[List[float]]:
    """
    Create embeddings for a list of texts with progress tracking.
    
    Args:
        texts: List of texts to create embeddings for
        model: OpenAI embedding model to use
        batch_size: Number of texts to process in each batch
        
    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    if not texts:
        return []
    
    def process_batch(batch: List[str]) -> List[List[float]]:
        return call_openai_embeddings(batch, model=model)
    
    return batch_process_with_progress(
        items=texts,
        process_func=process_batch,
        batch_size=batch_size,
        progress_desc=f"Creating embeddings"
    )


def generate_completions_with_progress(
    prompts: List[List[Dict[str, str]]],
    model: str = None,
    temperature: float = 0.3,
    max_tokens: int = 150,
    batch_size: int = 10
) -> List[str]:
    """
    Generate chat completions for a list of prompts with progress tracking.
    
    Args:
        prompts: List of message lists for chat completions
        model: OpenAI model to use
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        batch_size: Number of prompts to process in each batch
        
    Returns:
        List of generated text responses
    """
    if not prompts:
        return []
    
    def process_batch(batch: List[List[Dict[str, str]]]) -> List[str]:
        results = []
        for messages in batch:
            result = call_openai_chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            results.append(result)
        return results
    
    return batch_process_with_progress(
        items=prompts,
        process_func=process_batch,
        batch_size=batch_size,
        progress_desc="Generating completions"
    )


def print_error_summary() -> None:
    """Print a summary of errors encountered during API calls."""
    if error_counter:
        logger.info("=== OpenAI API Error Summary ===")
        for error_type, count in error_counter.most_common():
            logger.info(f"{error_type}: {count} occurrences")
        logger.info(f"Total retry attempts: {sum(error_counter.values())}")
        logger.info("===============================")
    else:
        logger.info("No OpenAI API errors encountered.")


def reset_error_counter() -> None:
    """Reset the error counter for fresh tracking."""
    global error_counter
    error_counter.clear()


# Async versions for future extensibility
async def call_openai_embeddings_async(
    texts: List[str], 
    model: str = "text-embedding-3-small"
) -> List[List[float]]:
    """
    Async version of embeddings creation (placeholder for future implementation).
    
    Note: Currently runs synchronously. True async implementation would require
    the async OpenAI client.
    """
    # For now, run the sync version in an executor
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, call_openai_embeddings, texts, model)


async def call_openai_chat_completion_async(
    messages: List[Dict[str, str]], 
    model: str = None,
    temperature: float = 0.3,
    max_tokens: int = 150
) -> str:
    """
    Async version of chat completion (placeholder for future implementation).
    
    Note: Currently runs synchronously. True async implementation would require
    the async OpenAI client.
    """
    # For now, run the sync version in an executor
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, 
        call_openai_chat_completion, 
        messages, 
        model, 
        temperature, 
        max_tokens
    )


# Decorator for easy function wrapping
def with_openai_retries(func: Callable) -> Callable:
    """
    Decorator to add retry logic to any function that makes OpenAI API calls.
    
    Usage:
        @with_openai_retries
        def my_openai_function():
            return openai.embeddings.create(...)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        @retry_openai_call
        def inner():
            try:
                return func(*args, **kwargs)
            except Exception as e:
                track_error(e)
                raise
        
        return inner()
    
    return wrapper


# Example usage and demo
def demo_usage():
    """
    Demo function showing how to use the OpenAI wrapper.
    
    Example scenario: Processing 100 content chunks for embedding.
    """
    print("=== OpenAI Wrapper Demo ===")
    
    # Reset error tracking
    reset_error_counter()
    
    # Example: Create embeddings for a list of texts
    sample_texts = [
        f"This is sample text number {i} for embedding demonstration."
        for i in range(20)
    ]
    
    print(f"Creating embeddings for {len(sample_texts)} texts...")
    embeddings = create_embeddings_with_progress(sample_texts)
    print(f"Successfully created {len(embeddings)} embeddings")
    
    # Example: Generate completions for prompts
    sample_prompts = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Summarize this: Sample content {i}"}
        ]
        for i in range(5)
    ]
    
    print(f"Generating completions for {len(sample_prompts)} prompts...")
    completions = generate_completions_with_progress(sample_prompts)
    print(f"Successfully generated {len(completions)} completions")
    
    # Print error summary
    print_error_summary()
    
    return embeddings, completions


if __name__ == "__main__":
    # Run demo if script is executed directly
    demo_usage()