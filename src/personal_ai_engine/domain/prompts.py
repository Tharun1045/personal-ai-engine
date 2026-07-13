# Prompts for RAG, contextual chunking, QA dataset generation, and document scoring.

RAG_SYSTEM_PROMPT = """You are an expert personal AI assistant. Use the following context retrieved from the user's second brain to answer their question.
If the context does not contain the information needed, say so. Do not make up answers.
Cite your sources by mentioning the title and URL of the document where possible.

Context:
{context}
"""

RAG_USER_PROMPT = "Question: {question}"

CONTEXTUAL_CHUNK_PROMPT = """<document>
{document_content}
</document>

Here is the chunk we want to contextualize:
<chunk>
{chunk_content}
</chunk>

Please provide a concise 1-2 sentence prefix to prepend to this chunk so that it is self-contained. The prefix should explain what the document is about and how this chunk fits in.
Only return the prefix, nothing else.
"""

QA_GENERATION_PROMPT = """You are an expert at generating high-quality training examples for fine-tuning an AI assistant.
Given the following DOCUMENT, generate exactly {num_pairs} distinct questions and answers pairs based on its content.
Each pair should contain:
1. A question that a user might ask about this document's content.
2. A detailed and accurate answer directly derived from the document.

The output must be formatted as a valid JSON list of objects:
[
  {{
    "instruction": "The question here",
    "output": "The answer here"
  }}
]

DOCUMENT:
{document}
"""

QUALITY_SCORING_PROMPT = """You are an expert judge tasked with evaluating the quality of a given DOCUMENT.

Guidelines:
1. Evaluate the DOCUMENT based on generally accepted facts and reliable information.
2. Evaluate that the DOCUMENT contains relevant information and not only links or error messages.
3. Check that the DOCUMENT doesn't oversimplify or generalize information in a way that changes its meaning or accuracy.

Analyze the text thoroughly and assign a quality score between 0 and 1, where:
- **0.0**: completely irrelevant containing only noise
- **0.1 - 0.7**: partially relevant
- **0.8 - 1.0**: entirely relevant

It is crucial that you return only the score in the following JSON format:
{{
    "score": <your score between 0.0 and 1.0>,
    "reasons": ["reason 1", "reason 2"]
}}

DOCUMENT:
{document}
"""
