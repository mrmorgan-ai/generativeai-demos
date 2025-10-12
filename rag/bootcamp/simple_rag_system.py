# Simple RAG System Tutorial
# Let's build a smart document search system step by step!

import PyPDF2
import re
import math
from collections import Counter
import numpy as np

# ========================================
# STEP 1: READING PDF FILES
# ========================================
# Think of this like opening a book and reading each page

def read_pdf_file(pdf_path):
    """
    This function opens a PDF file and reads all the text from it.
    It's like having a robot that can read every page of a book for you!
    """
    text = ""
    
    # Open the PDF file (like opening a book)
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        
        # Read each page (like turning pages in a book)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            # Extract text from this page
            page_text = page.extract_text()
            text += page_text + " "
    
    return text.strip()

# ========================================
# STEP 2: BREAKING TEXT INTO CHUNKS
# ========================================
# Imagine cutting a long story into smaller paragraphs

def create_chunks(text, chunk_size=200):
    """
    This breaks our long text into smaller pieces (chunks).
    Why? Because it's easier to find specific information in small pieces
    than in one giant block of text!
    
    Think of it like organizing your toys into different boxes
    instead of having one huge pile.
    """
    # Split text into words (like separating LEGO pieces)
    words = text.split()
    chunks = []
    
    # Group words into chunks of specified size
    for i in range(0, len(words), chunk_size):
        # Take a group of words (like grabbing a handful of LEGO pieces)
        chunk_words = words[i:i + chunk_size]
        # Join them back into a sentence
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
    
    return chunks

# ========================================
# STEP 3: SIMPLE TOKENIZATION
# ========================================
# This is like breaking sentences into individual words

def simple_tokenize(text):
    """
    Tokenization is like taking a sentence and separating each word.
    For example: "The cat sat" becomes ["the", "cat", "sat"]
    """
    # Convert to lowercase (so "Cat" and "cat" are treated the same)
    text = text.lower()
    # Remove punctuation and split into words
    words = re.findall(r'\b\w+\b', text)
    return words

# ========================================
# STEP 4: CREATING SIMPLE EMBEDDINGS
# ========================================
# This turns words into numbers that computers can understand

def create_vocabulary(chunks):
    """
    Create a vocabulary (dictionary) of all unique words.
    This is like making a list of every different word we've seen.
    """
    all_words = set()
    for chunk in chunks:
        words = simple_tokenize(chunk)
        all_words.update(words)
    
    # Convert to a sorted list and create word-to-index mapping
    vocab = sorted(list(all_words))
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    return vocab, word_to_idx

def text_to_vector(text, word_to_idx):
    """
    Convert text into a vector (list of numbers).
    This is like giving each word a unique number ID, then counting
    how many times each word appears in our text.
    
    Think of it like counting different colored marbles in a bag.
    """
    words = simple_tokenize(text)
    # Create a vector filled with zeros
    vector = [0] * len(word_to_idx)
    
    # Count how many times each word appears
    word_counts = Counter(words)
    
    # Fill in the vector with word counts
    for word, count in word_counts.items():
        if word in word_to_idx:
            idx = word_to_idx[word]
            vector[idx] = count
    
    return vector

def calculate_tf_idf(chunks, word_to_idx):
    """
    TF-IDF makes common words (like "the", "and") less important
    and rare, meaningful words more important.
    
    Think of it like this: if everyone has the same toy, it's not special.
    But if only you have a rare toy, that makes it very special!
    """
    # Convert all chunks to vectors
    vectors = []
    for chunk in chunks:
        vector = text_to_vector(chunk, word_to_idx)
        vectors.append(vector)
    
    # Calculate TF-IDF for each chunk
    tfidf_vectors = []
    num_docs = len(chunks)
    
    for doc_idx, vector in enumerate(vectors):
        tfidf_vector = []
        
        for word_idx, tf in enumerate(vector):
            if tf > 0:
                # Calculate how many documents contain this word
                df = sum(1 for v in vectors if v[word_idx] > 0)
                # Calculate TF-IDF score
                idf = math.log(num_docs / df)
                tfidf = tf * idf
            else:
                tfidf = 0
            
            tfidf_vector.append(tfidf)
        
        tfidf_vectors.append(tfidf_vector)
    
    return tfidf_vectors

# ========================================
# STEP 5: SIMPLE VECTOR DATABASE
# ========================================
# This is like organizing our number-vectors in a special filing cabinet

class SimpleVectorDB:
    """
    Our simple vector database is like a smart filing cabinet
    that can quickly find similar documents.
    """
    
    def __init__(self):
        self.vectors = []      # Store our number-vectors
        self.chunks = []       # Store the original text chunks
        self.word_to_idx = {}  # Our word dictionary
    
    def add_documents(self, chunks):
        """
        Add documents to our database.
        This is like filing papers in our smart cabinet.
        """
        print(f"Adding {len(chunks)} chunks to the database...")
        
        # Create vocabulary from all chunks
        vocab, self.word_to_idx = create_vocabulary(chunks)
        print(f"Created vocabulary with {len(vocab)} unique words")
        
        # Convert chunks to TF-IDF vectors
        self.vectors = calculate_tf_idf(chunks, self.word_to_idx)
        self.chunks = chunks
        
        print("Database ready!")
    
    def cosine_similarity(self, vec1, vec2):
        """
        Calculate how similar two vectors are.
        This is like measuring how similar two recipes are
        by comparing their ingredients.
        """
        # Convert to numpy arrays for easier calculation
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        # Calculate dot product
        dot_product = np.dot(vec1, vec2)
        
        # Calculate magnitudes (like measuring the length of vectors)
        magnitude1 = np.sqrt(np.sum(vec1**2))
        magnitude2 = np.sqrt(np.sum(vec2**2))
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        # Calculate cosine similarity
        similarity = dot_product / (magnitude1 * magnitude2)
        return similarity
    
    def search(self, query, top_k=3):
        """
        Search for the most relevant chunks for a given query.
        This is like asking our smart librarian to find the best books
        for our question.
        """
        print(f"Searching for: '{query}'")
        
        # Convert query to vector using the same process
        query_vector = text_to_vector(query, self.word_to_idx)
        
        # Calculate TF-IDF for query
        query_tf_idf = []
        num_docs = len(self.chunks)
        
        for word_idx, tf in enumerate(query_vector):
            if tf > 0:
                df = sum(1 for v in self.vectors if v[word_idx] > 0)
                if df > 0:
                    idf = math.log(num_docs / df)
                    tfidf = tf * idf
                else:
                    tfidf = 0
            else:
                tfidf = 0
            query_tf_idf.append(tfidf)
        
        # Find similarities with all chunks
        similarities = []
        for idx, chunk_vector in enumerate(self.vectors):
            similarity = self.cosine_similarity(query_tf_idf, chunk_vector)
            similarities.append((similarity, idx))
        
        # Sort by similarity (highest first) and get top results
        similarities.sort(reverse=True)
        top_results = similarities[:top_k]
        
        # Return the most relevant chunks
        results = []
        for similarity, idx in top_results:
            results.append({
                'chunk': self.chunks[idx],
                'similarity': similarity,
                'chunk_id': idx
            })
        
        return results

# ========================================
# STEP 6: PUTTING IT ALL TOGETHER
# ========================================
# Now let's use our RAG system!

def create_rag_system(pdf_path):
    """
    This creates our complete RAG system from a PDF file.
    It's like building a smart robot that can answer questions
    about any book you give it!
    """
    print("🤖 Building your RAG system...")
    
    # Step 1: Read the PDF
    print("\n📖 Step 1: Reading PDF file...")
    text = read_pdf_file(pdf_path)
    print(f"Read {len(text)} characters from PDF")
    
    # Step 2: Create chunks
    print("\n✂️ Step 2: Breaking text into chunks...")
    chunks = create_chunks(text, chunk_size=150)
    print(f"Created {len(chunks)} chunks")
    
    # Step 3: Create vector database
    print("\n🗃️ Step 3: Creating vector database...")
    vector_db = SimpleVectorDB()
    vector_db.add_documents(chunks)
    
    print("\n✅ RAG system ready! You can now ask questions.")
    return vector_db

def ask_question(vector_db, question):
    """
    Ask a question to our RAG system and get the best answer.
    """
    print(f"\n❓ Question: {question}")
    print("-" * 50)
    
    # Search for relevant chunks
    results = vector_db.search(question, top_k=3)
    
    # Display results
    for i, result in enumerate(results, 1):
        print(f"\n📄 Result {i} (Similarity: {result['similarity']:.3f}):")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Text: {result['chunk'][:300]}...")  # Show first 300 characters
        print()
    
    return results

# ========================================
# EXAMPLE USAGE
# ========================================

if __name__ == "__main__":
    # Example of how to use the system
    print("Welcome to the Simple RAG System Tutorial!")
    print("=" * 50)
    
    # For demonstration, let's create some sample text
    sample_text = """
    Artificial Intelligence is a fascinating field of computer science that focuses on creating 
    machines capable of intelligent behavior. Machine learning is a subset of AI that enables 
    computers to learn and improve from experience without being explicitly programmed.
    
    Deep learning is a specialized form of machine learning that uses neural networks with 
    multiple layers to model and understand complex patterns in data. These neural networks 
    are inspired by the human brain's structure and function.
    
    Natural Language Processing (NLP) is another important area of AI that deals with the 
    interaction between computers and human language. It enables machines to understand, 
    interpret, and generate human language in a valuable way.
    
    Computer vision is the field of AI that trains computers to see and understand the 
    visual world. Using digital images from cameras and videos, machines can accurately 
    identify and classify objects and then react to what they see.
    """
    
    # Simulate the process with sample text instead of PDF
    print("Using sample text for demonstration...")
    
    # Create chunks
    chunks = create_chunks(sample_text, chunk_size=50)
    
    # Create and populate vector database
    vector_db = SimpleVectorDB()
    vector_db.add_documents(chunks)
    
    # Example questions
    questions = [
        "What is machine learning?",
        "How do neural networks work?",
        "What is computer vision?"
    ]
    
    # Ask questions
    for question in questions:
        results = ask_question(vector_db, question)
    
    print("\n🎉 Tutorial complete! You now understand how RAG systems work!")
    print("\nTo use with real PDFs, replace the sample text section with:")
    print("vector_db = create_rag_system('your_document.pdf')")
