#!/usr/bin/env python3
"""
Demo script for the robust OpenAI wrapper.

This script demonstrates how to use the OpenAI wrapper for:
1. Batch embedding creation with progress tracking
2. Chat completions with retry logic
3. Error tracking and reporting

Example use case: Processing content chunks from a crawled website.
"""

import os
import sys
import logging
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai_wrapper import (
    create_embeddings_with_progress,
    call_openai_chat_completion,
    generate_completions_with_progress,
    print_error_summary,
    reset_error_counter,
    with_openai_retries
)

# Configure logging to show wrapper output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def demo_embedding_creation():
    """
    Demo: Creating embeddings for website content chunks.
    
    Simulates processing 100 content chunks from docs.api.com
    """
    print("\n" + "="*60)
    print("🚀 DEMO: Embedding Creation with Progress Tracking")
    print("="*60)
    
    # Reset error tracking for clean demo
    reset_error_counter()
    
    # Simulate 100 content chunks from a crawled website
    content_chunks = [
        f"API Documentation Chapter {i}: This section covers the implementation "
        f"details for feature {i} of the API. It includes code examples, "
        f"parameters, response formats, and common use cases that developers "
        f"should be aware of when integrating this functionality."
        for i in range(1, 101)
    ]
    
    print(f"📄 Processing {len(content_chunks)} content chunks from docs.api.com")
    print("💡 Features demonstrated:")
    print("   ✓ Progress bar with live updates")
    print("   ✓ Automatic retry on API failures")
    print("   ✓ Batch processing for efficiency")
    print("   ✓ Error tracking and reporting")
    
    try:
        # Create embeddings with progress tracking
        embeddings = create_embeddings_with_progress(
            texts=content_chunks,
            model="text-embedding-3-small",
            batch_size=20  # Process 20 at a time
        )
        
        print(f"✅ Successfully created embeddings for {len(embeddings)} chunks")
        print(f"📊 Embedding dimensions: {len(embeddings[0]) if embeddings else 0}")
        
    except Exception as e:
        print(f"❌ Error during embedding creation: {e}")
    
    # Show error summary
    print("\n📈 Error Summary:")
    print_error_summary()


def demo_chat_completions():
    """
    Demo: Generating summaries with chat completions.
    
    Simulates generating summaries for different types of content.
    """
    print("\n" + "="*60)
    print("🤖 DEMO: Chat Completions with Retry Logic")
    print("="*60)
    
    # Reset error tracking
    reset_error_counter()
    
    # Sample content for summarization
    content_samples = [
        {
            "type": "API Guide",
            "content": "This comprehensive guide covers the REST API endpoints for user management, authentication, and data retrieval. It includes examples in multiple programming languages and covers rate limiting, error handling, and best practices for integration."
        },
        {
            "type": "Code Tutorial",
            "content": "Learn how to build a real-time chat application using WebSockets and Node.js. This tutorial walks through setting up the server, handling connections, implementing message broadcasting, and adding user authentication."
        },
        {
            "type": "Configuration Manual",
            "content": "Complete configuration reference for the application deployment. Covers environment variables, database setup, SSL certificates, load balancing, monitoring, and troubleshooting common deployment issues."
        }
    ]
    
    print(f"📝 Generating summaries for {len(content_samples)} content types")
    print("💡 Features demonstrated:")
    print("   ✓ Individual chat completion calls")
    print("   ✓ Automatic retry with exponential backoff")
    print("   ✓ Error tracking per request")
    print("   ✓ Different content types handling")
    
    summaries = []
    
    for i, sample in enumerate(content_samples, 1):
        print(f"\n📋 Processing {sample['type']} ({i}/{len(content_samples)})")
        
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
                {"role": "user", "content": f"Create a 2-sentence summary of this {sample['type']}:\n\n{sample['content']}"}
            ]
            
            summary = call_openai_chat_completion(
                messages=messages,
                temperature=0.3,
                max_tokens=100
            )
            
            summaries.append({
                "type": sample["type"],
                "summary": summary
            })
            
            print(f"✅ Generated summary: {summary}")
            
        except Exception as e:
            print(f"❌ Error generating summary for {sample['type']}: {e}")
    
    print(f"\n📊 Successfully generated {len(summaries)} summaries")
    
    # Show error summary
    print("\n📈 Error Summary:")
    print_error_summary()


def demo_batch_completions():
    """
    Demo: Batch processing multiple completions with progress tracking.
    """
    print("\n" + "="*60)
    print("⚡ DEMO: Batch Completions with Progress Tracking")
    print("="*60)
    
    # Reset error tracking
    reset_error_counter()
    
    # Create multiple prompts for batch processing
    code_examples = [
        f"def process_data_{i}(data):\n    return data.transform().filter().sort()",
        f"class DataHandler_{i}:\n    def __init__(self):\n        self.data = []\n    def add(self, item):\n        self.data.append(item)",
        f"async def fetch_api_{i}(url):\n    async with session.get(url) as response:\n        return await response.json()",
        f"@decorator\ndef calculate_{i}(x, y):\n    result = x * y + 42\n    return result",
        f"for item in collection_{i}:\n    if item.is_valid():\n        process(item)\n    else:\n        log_error(item)"
    ]
    
    # Convert to message format for batch processing
    prompts = []
    for i, code in enumerate(code_examples):
        prompts.append([
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": f"Explain what this code does in one sentence:\n\n{code}"}
        ])
    
    print(f"🔍 Analyzing {len(code_examples)} code examples")
    print("💡 Features demonstrated:")
    print("   ✓ Batch processing with progress bar")
    print("   ✓ Error handling for individual items")
    print("   ✓ Configurable batch sizes")
    print("   ✓ Automatic retry for failed items")
    
    try:
        # Process completions in batches
        explanations = generate_completions_with_progress(
            prompts=prompts,
            temperature=0.2,
            max_tokens=50,
            batch_size=2  # Process 2 at a time for demo
        )
        
        print(f"\n📊 Generated {len(explanations)} code explanations:")
        for i, explanation in enumerate(explanations):
            print(f"   {i+1}. {explanation}")
            
    except Exception as e:
        print(f"❌ Error during batch processing: {e}")
    
    # Show error summary
    print("\n📈 Error Summary:")
    print_error_summary()


def demo_decorator_usage():
    """
    Demo: Using the @with_openai_retries decorator.
    """
    print("\n" + "="*60)
    print("🎯 DEMO: Decorator Usage for Custom Functions")
    print("="*60)
    
    @with_openai_retries
    def custom_embedding_function(text: str) -> List[float]:
        """Custom function that uses OpenAI API with automatic retries."""
        import openai
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=[text]
        )
        return response.data[0].embedding
    
    @with_openai_retries
    def custom_chat_function(prompt: str) -> str:
        """Custom function for chat completion with automatic retries."""
        import openai
        response = openai.chat.completions.create(
            model=os.getenv("MODEL_CHOICE", "gpt-3.5-turbo"),
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    
    print("🔧 Testing custom functions with retry decorator")
    print("💡 Features demonstrated:")
    print("   ✓ @with_openai_retries decorator")
    print("   ✓ Automatic error tracking")
    print("   ✓ Seamless integration with existing code")
    
    # Reset error tracking
    reset_error_counter()
    
    try:
        # Test custom embedding function
        text = "This is a test for the custom embedding function with retries."
        embedding = custom_embedding_function(text)
        print(f"✅ Custom embedding created: {len(embedding)} dimensions")
        
        # Test custom chat function
        prompt = "What is the capital of France?"
        response = custom_chat_function(prompt)
        print(f"✅ Custom chat response: {response}")
        
    except Exception as e:
        print(f"❌ Error in custom functions: {e}")
    
    # Show error summary
    print("\n📈 Error Summary:")
    print_error_summary()


def demo_real_world_scenario():
    """
    Demo: Complete real-world scenario.
    
    Simulates the full pipeline: crawl site → chunk content → embed → store.
    """
    print("\n" + "="*60)
    print("🌍 DEMO: Real-World Scenario - Complete Pipeline")
    print("="*60)
    
    # Reset error tracking for the full pipeline
    reset_error_counter()
    
    # Simulate crawled content from docs.api.com
    crawled_pages = [
        {
            "url": "https://docs.api.com/getting-started",
            "content": "Getting started with our API is simple. First, obtain your API key from the dashboard. Then, make your first request using any HTTP client. Our API supports both REST and GraphQL endpoints."
        },
        {
            "url": "https://docs.api.com/authentication", 
            "content": "Authentication is handled via API keys or OAuth 2.0. Include your API key in the Authorization header. For OAuth, follow the standard flow to obtain access tokens."
        },
        {
            "url": "https://docs.api.com/rate-limits",
            "content": "Rate limits apply to all API endpoints. Free tier allows 1000 requests per hour. Premium accounts get 10,000 requests per hour. Use the X-RateLimit headers to monitor your usage."
        }
    ]
    
    print(f"🕷️  Simulating crawl of docs.api.com ({len(crawled_pages)} pages)")
    print("💡 Complete pipeline demonstration:")
    print("   ✓ Content chunking")
    print("   ✓ Contextual embedding generation")
    print("   ✓ Batch embedding creation") 
    print("   ✓ Progress tracking throughout")
    print("   ✓ Comprehensive error reporting")
    
    # Step 1: Create content chunks
    chunks = []
    for page in crawled_pages:
        # Simple chunking by sentences (in real app, you'd use more sophisticated chunking)
        sentences = page["content"].split(". ")
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                chunks.append({
                    "url": page["url"],
                    "chunk_number": i,
                    "content": sentence.strip() + "." if not sentence.endswith(".") else sentence.strip()
                })
    
    print(f"\n📄 Created {len(chunks)} content chunks")
    
    # Step 2: Generate contextual embeddings for chunks
    try:
        chunk_texts = [chunk["content"] for chunk in chunks]
        
        print("\n🧠 Generating contextual information...")
        enhanced_chunks = []
        for chunk in chunks:
            # Find the full document for context
            full_doc = next(page["content"] for page in crawled_pages if page["url"] == chunk["url"])
            
            # Generate contextual embedding
            messages = [
                {"role": "system", "content": "Provide brief context for this content chunk."},
                {"role": "user", "content": f"Document: {full_doc}\n\nChunk: {chunk['content']}\n\nProvide 1 sentence of context for this chunk:"}
            ]
            
            try:
                context = call_openai_chat_completion(
                    messages=messages,
                    max_tokens=50,
                    temperature=0.3
                )
                enhanced_content = f"{context} {chunk['content']}"
                enhanced_chunks.append(enhanced_content)
            except Exception as e:
                logger.warning(f"Failed to generate context for chunk, using original: {e}")
                enhanced_chunks.append(chunk['content'])
        
        print(f"✅ Enhanced {len(enhanced_chunks)} chunks with context")
        
        # Step 3: Create embeddings for all enhanced chunks
        print("\n📊 Creating embeddings for enhanced chunks...")
        embeddings = create_embeddings_with_progress(
            texts=enhanced_chunks,
            model="text-embedding-3-small",
            batch_size=5
        )
        
        print(f"✅ Created {len(embeddings)} embeddings")
        print(f"📏 Average embedding dimension: {len(embeddings[0]) if embeddings else 0}")
        
        # Step 4: Simulate storage (in real app, you'd store to Supabase)
        print("\n💾 Simulating storage to vector database...")
        stored_count = 0
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if embedding and not all(v == 0.0 for v in embedding):
                stored_count += 1
        
        print(f"✅ Successfully stored {stored_count}/{len(chunks)} chunks with embeddings")
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
    
    # Final error summary for the entire pipeline
    print("\n📈 Complete Pipeline Error Summary:")
    print_error_summary()
    
    print("\n🎉 Real-world scenario demonstration completed!")


def main():
    """Run all demos."""
    print("🚀 OpenAI Wrapper Robust Retry & Progress Tracking Demo")
    print("=" * 70)
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set. Some demos may fail.")
        print("   Set your API key: export OPENAI_API_KEY='your-key-here'")
    
    # Check if MODEL_CHOICE is set
    model_choice = os.getenv("MODEL_CHOICE", "gpt-3.5-turbo")
    print(f"🤖 Using model: {model_choice}")
    
    try:
        # Run all demos
        demo_embedding_creation()
        demo_chat_completions()
        demo_batch_completions()
        demo_decorator_usage()
        demo_real_world_scenario()
        
        print("\n" + "="*70)
        print("✅ All demos completed successfully!")
        print("💡 Key benefits demonstrated:")
        print("   • Automatic retry with exponential backoff + jitter")
        print("   • Live progress bars for batch operations")
        print("   • Comprehensive error tracking and reporting")
        print("   • Easy integration with existing code")
        print("   • Rate limiting and robust error handling")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        print_error_summary()


if __name__ == "__main__":
    main()