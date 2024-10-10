from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Sample documents to retrieve from
documents = [
    "The Eiffel Tower is located in Paris.",
    "Python is a programming language used for AI and machine learning.",
    "The capital of France is Paris.",
    "Machine learning models can be used for image recognition.",
]

# SETP 1. RETRIEVAL: A simple TF-IDF-based retriever
def retrieve_relevant_documents(query, documents):
    """
    Retrieve the most relevant document from the corpus using TF-IDF.
    
    Parameters:
    - query (str): The user's query or question.
    - documents (list of str): A list of documents to retrieve from.

    Returns:
    - str: The most relevant document based on cosine similarity with the query.
    """
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(documents)
    
    # Transform the query into the same vector space
    query_vec = vectorizer.transform([query])
    
    # Compute cosine similarity between the query and the documents
    cosine_similarities = np.dot(doc_vectors.toarray(), query_vec.T.toarray())
    
    # Get the index of the most relevant document
    most_relevant_doc_index = np.argmax(cosine_similarities)
    return documents[most_relevant_doc_index]

# STEP 2. GENERATION: Using GPT-2 for text generation
def generate_response(retrieved_doc, query):
    """
    Generate a natural language response using GPT-2 based on the retrieved document and query.
    
    Parameters:
    - retrieved_doc (str): The document retrieved from the retrieval step.
    - query (str): The user's query.

    Returns:
    - str: A generated response that answers the query based on the retrieved document.
    """
    # Load pre-trained GPT-2 model and tokenizer
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Prepare the input for GPT-2
    input_text = f"Context: {retrieved_doc}\nQuestion: {query}\nAnswer:"
    inputs = tokenizer.encode(input_text, return_tensors="pt")

    # Generate a response from GPT-2
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text

# STEP 3. RAG (Retrieval-Augmented Generation) System combining retrieval and generation)
def RAG_system(query, documents):
    """
    RAG (Retrieval-Augmented Generation) system that retrieves a relevant document based on a query
    and generates a natural language response using GPT-2.
    
    Parameters:
    - query (str): The user's input query.
    - documents (list of str): A list of documents to retrieve from.

    Returns:
    - str: A generated response that uses the retrieved document as context.
    """
    retrieved_doc = retrieve_relevant_documents(query, documents)
    print(f"Retrieved Document: {retrieved_doc}")
    
    response = generate_response(retrieved_doc, query)
    return response

# Example query & answer 
query = "Where is the Eiffel Tower?"
answer = RAG_system(query, documents)
print(f"\nGenerated Response: {answer}")
