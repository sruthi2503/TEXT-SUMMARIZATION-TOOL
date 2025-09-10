from transformers import pipeline

def summarize_text(text, max_length=130, min_length=30):
    # Load summarization pipeline
    summarizer = pipeline("summarization")

    # Generate summary
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

if __name__ == "__main__":
    # Example lengthy article text
    article_text = """
    Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. The ultimate objective of NLP is to read, decipher, understand, and make sense of human languages in a manner that is valuable. Many challenges in NLP involve natural language understanding, natural language generation, and machine translation. Recent advances in deep learning have significantly improved the performance of NLP systems, enabling applications such as chatbots, automatic summarization, sentiment analysis, and more.
    """

    summary = summarize_text(article_text)
    print("Original Text:\n", article_text)
    print("\nSummary:\n", summary)
