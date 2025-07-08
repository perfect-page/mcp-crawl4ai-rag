"""
OpenAI API Wrapper with Rate Limiting, Retries, and Progress Tracking

This module provides a robust wrapper for OpenAI API calls with:
- Proactive rate limiting
- Automatic retry with exponential backoff
- Live progress bar with error counters
- Comprehensive error tracking and logging
- Reusable design for both embeddings and chat completions
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional, Callable, Union
from collections import Counter
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
from pyrate_limiter import Duration, Limiter, RequestRate
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global error counter for tracking errors across all calls
error_counter = Counter()

# Rate limiter configuration (can be overridden via environment variables)
DEFAULT_RPM_LIMIT = int(os.getenv("OPENAI_RPM_LIMIT", "60"))
DEFAULT_TOKENS_PER_MINUTE = int(os.getenv("OPENAI_TPM_LIMIT", "150000"))

# Create rate limiter instance
rate_limiter = Limiter(
    RequestRate(DEFAULT_RPM_LIMIT, Duration.MINUTE),
    RequestRate(DEFAULT_TOKENS_PER_MINUTE // 1000, Duration.MINUTE)  # Rough token estimate
)


class OpenAIWrapper:
    """
    A robust wrapper for OpenAI API calls with rate limiting, retries, and progress tracking.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        rpm_limit: Optional[int] = None,
        show_progress: bool = True,
        max_parallel_calls: int = 10
    ):
        """
        Initialize the OpenAI wrapper.
        
        Args:
            api_key: OpenAI API key (uses env var if not provided)
            rpm_limit: Requests per minute limit (uses env var if not provided)
            show_progress: Whether to show progress bars
            max_parallel_calls: Maximum number of parallel API calls
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY env var")
        
        openai.api_key = self.api_key
        
        # Configure rate limiter
        if rpm_limit:
            self.rate_limiter = Limiter(
                RequestRate(rpm_limit, Duration.MINUTE),
                RequestRate(rpm_limit * 2500, Duration.MINUTE)  # Token estimate
            )
        else:
            self.rate_limiter = rate_limiter
        
        self.show_progress = show_progress
        self.max_parallel_calls = max_parallel_calls
        self.call_counter = 0
        self.error_counter = Counter()
    
    def _extract_error_info(self, exception: Exception) -> tuple[str, str]:
        """Extract error code and message from exception."""
        if hasattr(exception, 'http_status'):
            code = str(exception.http_status)
        elif hasattr(exception, 'status_code'):
            code = str(exception.status_code)
        elif hasattr(exception, 'code'):
            code = str(exception.code)
        else:
            code = type(exception).__name__
        
        message = str(exception)
        return code, message
    
    @retry(
        stop=stop_never,  # Keep retrying until success
        wait=wait_exponential(multiplier=1, min=1, max=60),  # Exponential backoff with jitter
        retry=retry_if_exception_type((
            openai.RateLimitError,
            openai.APIError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
            Exception  # Catch all for safety
        )),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO)
    )
    def _call_with_retry(self, api_call: Callable, *args, **kwargs) -> Any:
        """
        Execute an API call with retry logic and error tracking.
        
        Args:
            api_call: The API call function to execute
            *args: Positional arguments for the API call
            **kwargs: Keyword arguments for the API call
            
        Returns:
            The API response
        """
        try:
            # Apply rate limiting
            with self.rate_limiter.ratelimit("openai", delay=True):
                result = api_call(*args, **kwargs)
                self.call_counter += 1
                return result
        except Exception as e:
            code, message = self._extract_error_info(e)
            self.error_counter[code] += 1
            error_counter[code] += 1  # Update global counter
            logger.warning(f"API call failed with {code}: {message}")
            raise  # Re-raise for tenacity to handle
    
    def create_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """
        Create a single embedding with retry logic.
        
        Args:
            text: Text to create embedding for
            model: OpenAI embedding model to use
            
        Returns:
            List of floats representing the embedding
        """
        response = self._call_with_retry(
            openai.embeddings.create,
            model=model,
            input=[text]
        )
        return response.data[0].embedding
    
    def create_embeddings_batch(
        self, 
        texts: List[str], 
        model: str = "text-embedding-3-small",
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Create embeddings for multiple texts with progress tracking.
        
        Args:
            texts: List of texts to create embeddings for
            model: OpenAI embedding model to use
            batch_size: Number of texts to process in each API call
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        embeddings = []
        total = len(texts)
        
        # Create progress bar
        pbar = tqdm(
            total=total, 
            desc="Creating embeddings",
            disable=not self.show_progress
        )
        
        # Process in batches
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = self._call_with_retry(
                    openai.embeddings.create,
                    model=model,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Failed to create embeddings for batch {i//batch_size + 1}: {e}")
                # Add zero embeddings as fallback
                embeddings.extend([[0.0] * 1536 for _ in batch])
            
            # Update progress bar
            pbar.update(len(batch))
            pbar.set_postfix({
                'completed': self.call_counter,
                **{f'err_{k}': v for k, v in self.error_counter.most_common(5)}
            })
        
        pbar.close()
        return embeddings
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Create a chat completion with retry logic.
        
        Args:
            messages: List of message dictionaries
            model: OpenAI model to use (uses MODEL_CHOICE env var if not provided)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Additional arguments for the API call
            
        Returns:
            The assistant's response content
        """
        model = model or os.getenv("MODEL_CHOICE", "gpt-4")
        
        response = self._call_with_retry(
            openai.chat.completions.create,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return response.choices[0].message.content
    
    def process_batch_with_progress(
        self,
        items: List[Any],
        process_func: Callable[[Any], Any],
        desc: str = "Processing",
        parallel: bool = True
    ) -> List[Any]:
        """
        Process a batch of items with progress tracking and parallel execution.
        
        Args:
            items: List of items to process
            process_func: Function to process each item (should use the wrapper's methods)
            desc: Description for the progress bar
            parallel: Whether to process items in parallel
            
        Returns:
            List of results in the same order as input items
        """
        if not items:
            return []
        
        results = [None] * len(items)
        total = len(items)
        
        # Create progress bar
        pbar = tqdm(
            total=total,
            desc=desc,
            disable=not self.show_progress
        )
        
        if parallel and self.max_parallel_calls > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.max_parallel_calls) as executor:
                # Submit all tasks
                future_to_idx = {
                    executor.submit(process_func, item): idx
                    for idx, item in enumerate(items)
                }
                
                # Process completed tasks
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        logger.error(f"Failed to process item {idx}: {e}")
                        results[idx] = None
                    
                    # Update progress
                    pbar.update(1)
                    pbar.set_postfix({
                        'completed': self.call_counter,
                        **{f'err_{k}': v for k, v in self.error_counter.most_common(5)}
                    })
        else:
            # Sequential processing
            for idx, item in enumerate(items):
                try:
                    results[idx] = process_func(item)
                except Exception as e:
                    logger.error(f"Failed to process item {idx}: {e}")
                    results[idx] = None
                
                # Update progress
                pbar.update(1)
                pbar.set_postfix({
                    'completed': self.call_counter,
                    **{f'err_{k}': v for k, v in self.error_counter.most_common(5)}
                })
        
        pbar.close()
        return results
    
    def print_error_summary(self):
        """Print a summary of all errors encountered."""
        if not self.error_counter:
            print("\n✅ No errors encountered!")
            return
        
        print("\n📊 Error Summary:")
        print("-" * 40)
        for code, count in self.error_counter.most_common():
            print(f"  {code}: {count} error{'s' if count > 1 else ''}")
        print("-" * 40)
        print(f"Total API calls: {self.call_counter}")
        print(f"Success rate: {(self.call_counter / (self.call_counter + sum(self.error_counter.values())) * 100):.1f}%")


# Convenience functions for backward compatibility
def call_openai_with_retries(
    prompt: str,
    api_type: str = "chat",
    model: Optional[str] = None,
    **kwargs
) -> Union[str, List[float]]:
    """
    Convenience function for single API calls with retries.
    
    Args:
        prompt: The prompt or text to process
        api_type: Either "chat" or "embedding"
        model: Model to use (uses env defaults if not provided)
        **kwargs: Additional arguments for the API call
        
    Returns:
        Response content (str for chat, List[float] for embedding)
    """
    wrapper = OpenAIWrapper(show_progress=False)
    
    if api_type == "embedding":
        return wrapper.create_embedding(prompt, model=model or "text-embedding-3-small")
    elif api_type == "chat":
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        return wrapper.chat_completion(messages, model=model, **kwargs)
    else:
        raise ValueError(f"Unknown api_type: {api_type}")


# Decorator for adding retry logic to existing functions
def with_openai_retries(show_progress: bool = True):
    """
    Decorator to add OpenAI retry logic to a function.
    
    The decorated function should accept an OpenAIWrapper instance as its first argument.
    
    Args:
        show_progress: Whether to show progress bars
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create wrapper instance
            openai_wrapper = OpenAIWrapper(show_progress=show_progress)
            
            # Call the function with the wrapper
            try:
                result = func(openai_wrapper, *args, **kwargs)
                
                # Print error summary if there were errors
                if openai_wrapper.error_counter and show_progress:
                    openai_wrapper.print_error_summary()
                
                return result
            except Exception as e:
                logger.error(f"Function {func.__name__} failed: {e}")
                if show_progress:
                    openai_wrapper.print_error_summary()
                raise
        
        return wrapper
    return decorator


# Example usage functions
if __name__ == "__main__":
    # Example 1: Creating embeddings with progress
    texts = ["Hello world", "How are you?", "OpenAI is great"] * 10
    wrapper = OpenAIWrapper()
    
    print("Creating embeddings...")
    embeddings = wrapper.create_embeddings_batch(texts)
    wrapper.print_error_summary()
    
    # Example 2: Using the decorator
    @with_openai_retries(show_progress=True)
    def process_documents(wrapper: OpenAIWrapper, documents: List[str]) -> List[str]:
        """Process documents to generate summaries."""
        def summarize(doc):
            return wrapper.chat_completion(
                messages=[
                    {"role": "system", "content": "Summarize the following text in one sentence."},
                    {"role": "user", "content": doc}
                ],
                max_tokens=50
            )
        
        return wrapper.process_batch_with_progress(
            documents,
            summarize,
            desc="Generating summaries"
        )
    
    # Example 3: Using convenience function
    response = call_openai_with_retries(
        "What is the capital of France?",
        api_type="chat"
    )
    print(f"Response: {response}")