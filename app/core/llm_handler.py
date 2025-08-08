# smartfile_ai/app/core/llm_handler.py

import os
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# MODEL_ID = "google/flan-t5-large"  # bigger, slower, better quality
MODEL_ID = "google/flan-t5-base"    # smaller, faster for testing

def get_llm():
    """
    Initializes and returns a Hugging Face LLM pipeline for FLAN-T5.
    """
    print(f"Loading Hugging Face model: {MODEL_ID}. This may take a few minutes on the first run.")

    # Load tokenizer and model for seq2seq
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, device_map="auto")

    # Create a text2text generation pipeline (for encoder-decoder models like T5)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
    )

    # Wrap in LangChain LLM
    return HuggingFacePipeline(pipeline=pipe)

def get_prompt_template():
    """
    Defines and returns the prompt template for the LLM.
    """
    prompt_template = """Use the following pieces of context to answer the question.
    Context: {context}
    Question: {question}
    Answer:"""
    
    return PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

# --- Test Run ---
if __name__ == "__main__":
    print("--- Running llm_handler.py test ---")
    llm = get_llm()
    context = "The capital of France is Paris. The Eiffel Tower is a famous landmark there. The official language is French."
    question = "What is the capital of France and its official language?"

    prompt = get_prompt_template()
    formatted_prompt = prompt.format(context=context, question=question)

    print("\n--- Prompt ---")
    print(formatted_prompt)

    response = llm.invoke(formatted_prompt)
    print("\n--- Response ---")
    print(response)
