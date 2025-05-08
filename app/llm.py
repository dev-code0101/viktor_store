import openai

# import os
import logging
from typing import List

from .config import LLM_MODEL_NAME, GROQ_API_KEY  # ,OPENAI_API_KEY
from .models import Chunk
from groq import Groq

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure DeepAI API key
# openai.api_key = OPENAI_API_KEY
groq = Groq(
    api_key=GROQ_API_KEY,
)


def get_llm_answer(
    question: str, context_chunks: List[Chunk], model_name: str = LLM_MODEL_NAME
) -> str:
    """Gets an answer from the LLM using the provided context."""
    # if not openai.api_key:
    #     logger.error("DeepAI API key is not set.")
    #     return "Error: DeepAI API key not configured."

    context = "\n\n".join([chunk.text for chunk in context_chunks])

    # Using Chat Completions API which is recommended
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions based on the provided context.",
        },
        {
            "role": "user",
            "content": f"Given the following context:\n{context}\n\nAnswer the following question:\n{question}",
        },
    ]

    try:
        # response = openai.ChatCompletion.create(
        #     model=model_name,
        #     messages=messages,
        #     max_tokens=300,  # Increased max tokens for potentially longer answers
        #     temperature=0.5,
        # )
        response = groq.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_completion_tokens=1024,
        )
        answer = response.choices[0].message.content.strip()
        logger.info(f"LLM generated answer for question: '{question[:50]}...'")
        return answer
    except openai.error.AuthenticationError:
        logger.error(
            "DeepAI Authentication Error: Invalid API key or organization."
        )
        return "Error: DeepAI Authentication failed. Please check your API key."
    except openai.error.APIError as e:
        logger.error(f"DeepAI API Error: {e}")
        return f"Error: DeepAI API error occurred: {e}"
    except Exception as e:
        logger.error(f"An unexpected error occurred during LLM call: {e}")
        return f"Error: An unexpected error occurred: {e}"


if __name__ == '__main__':
    # Example usage for testing the llm module
    # This requires a valid DeepAI API key and some dummy chunks
    if GROQ_API_KEY:
        dummy_chunks = [
            Chunk(
                doc="dummy.txt",
                chunk_id=0,
                text="The product has a one-year warranty.",
            ),
            Chunk(
                doc="dummy.txt",
                chunk_id=1,
                text="For support, please visit our website.",
            ),
        ]
        sample_question = "What is the warranty period?"
        answer = get_llm_answer(sample_question, dummy_chunks)
        print(f"\nQuestion: {sample_question}")
        print(f"LLM Answer: {answer}")

        sample_question_no_context = "What is the capital of France?"
        answer_no_context = get_llm_answer(
            sample_question_no_context, []
        )  # Test with no context
        print(f"\nQuestion: {sample_question_no_context}")
        print(f"LLM Answer (no context): {answer_no_context}")

    else:
        print("OPENAI_API_KEY is not set. Cannot run LLM module test.")
