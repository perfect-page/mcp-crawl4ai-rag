#!/usr/bin/env python3
"""
Demo script for the OpenAI wrapper with rate limiting, retries, and progress tracking.

This script demonstrates:
1. Creating embeddings with progress bar
2. Batch processing with error tracking
3. Using the decorator pattern
4. Handling failures gracefully
"""

import os
import sys
from typing import List
from openai_wrapper import OpenAIWrapper, call_openai_with_retries, with_openai_retries

# Ensure we have an API key
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set!")
    sys.exit(1)


def demo_single_calls():
    """Demonstrate single API calls with retries."""
    print("\n=== Demo: Single API Calls ===\n")
    
    # Example 1: Chat completion
    print("1. Chat completion example:")
    response = call_openai_with_retries(
        "What are the key benefits of using rate limiting in API clients?",
        api_type="chat",
        max_tokens=100
    )
    print(f"Response: {response}\n")
    
    # Example 2: Single embedding
    print("2. Single embedding example:")
    embedding = call_openai_with_retries(
        "OpenAI embeddings are useful for semantic search",
        api_type="embedding"
    )
    print(f"Embedding dimensions: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}\n")


def demo_batch_embeddings():
    """Demonstrate batch embedding creation with progress bar."""
    print("\n=== Demo: Batch Embeddings with Progress ===\n")
    
    # Create sample texts
    texts = [
        "Machine learning is transforming industries",
        "Natural language processing enables human-computer interaction",
        "Deep learning models require large datasets",
        "Transfer learning reduces training time",
        "Reinforcement learning optimizes decision making",
        "Computer vision interprets visual information",
        "Generative AI creates new content",
        "Federated learning preserves privacy",
        "Neural networks mimic brain structure",
        "Transformers revolutionized NLP"
    ] * 5  # Repeat to create 50 texts
    
    # Create wrapper instance
    wrapper = OpenAIWrapper(show_progress=True)
    
    print(f"Creating embeddings for {len(texts)} texts...")
    embeddings = wrapper.create_embeddings_batch(texts, batch_size=10)
    
    print(f"\nSuccessfully created {len(embeddings)} embeddings")
    wrapper.print_error_summary()


def demo_batch_processing_with_decorator():
    """Demonstrate batch processing using the decorator pattern."""
    print("\n=== Demo: Batch Processing with Decorator ===\n")
    
    @with_openai_retries(show_progress=True)
    def analyze_topics(wrapper: OpenAIWrapper, topics: List[str]) -> List[str]:
        """Analyze topics and generate descriptions."""
        def analyze_topic(topic):
            return wrapper.chat_completion(
                messages=[
                    {"role": "system", "content": "You are an expert that provides brief, insightful topic analysis."},
                    {"role": "user", "content": f"Provide a one-sentence insight about: {topic}"}
                ],
                max_tokens=50,
                temperature=0.7
            )
        
        return wrapper.process_batch_with_progress(
            topics,
            analyze_topic,
            desc="Analyzing topics",
            parallel=True
        )
    
    # Topics to analyze
    topics = [
        "Quantum computing",
        "Blockchain technology",
        "Artificial general intelligence",
        "Climate change solutions",
        "Space exploration",
        "Biotechnology advances",
        "Renewable energy",
        "Cybersecurity threats",
        "Edge computing",
        "5G networks"
    ]
    
    print(f"Analyzing {len(topics)} topics...")
    insights = analyze_topics(topics)
    
    print("\nTopic Insights:")
    for topic, insight in zip(topics, insights):
        if insight:
            print(f"- {topic}: {insight}")


def demo_error_handling():
    """Demonstrate error handling and recovery."""
    print("\n=== Demo: Error Handling ===\n")
    
    # Create wrapper with custom settings
    wrapper = OpenAIWrapper(
        rpm_limit=5,  # Very low limit to potentially trigger rate limiting
        show_progress=True
    )
    
    # Try to create many embeddings quickly
    texts = ["Test text " + str(i) for i in range(20)]
    
    print("Attempting rapid API calls with low rate limit...")
    print("(This will demonstrate retry logic and error tracking)\n")
    
    embeddings = wrapper.create_embeddings_batch(texts, batch_size=5)
    
    print(f"\nProcessed {len(embeddings)} embeddings")
    wrapper.print_error_summary()


def demo_contextual_processing():
    """Demonstrate contextual text processing."""
    print("\n=== Demo: Contextual Processing ===\n")
    
    @with_openai_retries(show_progress=True)
    def create_contextual_summaries(wrapper: OpenAIWrapper, documents: List[dict]) -> List[str]:
        """Create contextual summaries for document chunks."""
        def process_chunk(doc):
            prompt = f"""
Document: {doc['title']}
Section: {doc['section']}
Content: {doc['content']}

Create a brief contextual summary that explains what this section covers within the broader document.
"""
            return wrapper.chat_completion(
                messages=[
                    {"role": "system", "content": "You are a technical documentation expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.3
            )
        
        return wrapper.process_batch_with_progress(
            documents,
            process_chunk,
            desc="Creating contextual summaries"
        )
    
    # Sample documents
    documents = [
        {
            "title": "Python Best Practices Guide",
            "section": "Error Handling",
            "content": "Always use specific exception types rather than bare except clauses..."
        },
        {
            "title": "API Design Principles",
            "section": "Rate Limiting",
            "content": "Implement rate limiting to protect your API from abuse..."
        },
        {
            "title": "Machine Learning Fundamentals",
            "section": "Data Preprocessing",
            "content": "Clean and normalize your data before training models..."
        }
    ]
    
    summaries = create_contextual_summaries(documents)
    
    print("\nContextual Summaries:")
    for doc, summary in zip(documents, summaries):
        if summary:
            print(f"\n{doc['title']} - {doc['section']}:")
            print(f"  {summary}")


def main():
    """Run all demos."""
    print("🚀 OpenAI Wrapper Demo")
    print("=" * 50)
    
    # Run demos
    demo_single_calls()
    demo_batch_embeddings()
    demo_batch_processing_with_decorator()
    demo_contextual_processing()
    demo_error_handling()
    
    print("\n✅ All demos completed!")
    print("\nKey features demonstrated:")
    print("- ⚡ Automatic rate limiting")
    print("- 🔄 Retry with exponential backoff")
    print("- 📊 Live progress bars")
    print("- 🚨 Error tracking and reporting")
    print("- 🎯 Parallel processing support")
    print("- 🛡️ Graceful failure handling")


if __name__ == "__main__":
    main()