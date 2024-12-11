import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text(text):
    """
    Preprocess the text by tokenizing, removing stop words, and stemming.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize
    words = text.split()
    
    # Remove stop words (for simplicity, a basic stop-word list)
    stop_words = set(["and", "the", "is", "in", "to", "with", "a", "an", "of", "for", "on", "this", "that", "it", "by"])
    words = [word for word in words if word not in stop_words]
    
    return " ".join(words)

def read_documents(file_paths):
    """
    Read and preprocess documents from given file paths.
    """
    documents = {}
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as file:
            documents[os.path.basename(path)] = preprocess_text(file.read())
    return documents

def compute_similarity(documents):
    """
    Compute cosine similarity between documents.
    """
    # Convert documents to a TF-IDF matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents.values())
    
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    return similarity_matrix, list(documents.keys())

def display_results(similarity_matrix, filenames, result_label):
    """
    Display similarity results in a user-friendly manner.
    """
    results = "\nPlagiarism Detection Results:\n\n"
    found = False
    for i in range(len(filenames)):
        for j in range(i + 1, len(filenames)):
            similarity_score = similarity_matrix[i, j] * 100  # Convert to percentage
            if similarity_score > 30:  # Threshold for plagiarism
                results += f"{filenames[i]} and {filenames[j]} have a similarity score of {similarity_score:.2f}%\n"
                found = True
    
    if not found:
        results += "No significant similarities found."
    
    result_label.config(text=results)

def select_files():
    """
    Open a file dialog to select files.
    """
    file_paths = filedialog.askopenfilenames(filetypes=[("Text Files", "*.txt")])
    return file_paths
def analyze_documents():
    """
    Open a file dialog to select multiple files and compare them.
    """
    file_paths = select_files()  # Use the select_files function to choose files
    if not file_paths:
        messagebox.showerror("Error", "No files selected.")
        return

    documents = read_documents(file_paths)  # Read and preprocess the selected documents
    if len(documents) < 2:
        messagebox.showerror("Error", "At least two documents are required for comparison.")
        return

    # Compute similarity between the documents
    similarity_matrix, filenames = compute_similarity(documents)

    # Display the results
    display_results(similarity_matrix, filenames, result_label)

def select_two_files_and_compare():
    """
    Open a file dialog to select two files and compare them.
    """
    file_paths = filedialog.askopenfilenames(filetypes=[("Text Files", "*.txt")], title="Select Two Files")
    if len(file_paths) != 2:
        messagebox.showerror("Error", "Please select exactly two files.")
        return

    documents = read_documents(file_paths)
    if len(documents) != 2:
        messagebox.showerror("Error", "Failed to read two valid text documents.")
        return

    similarity_matrix, filenames = compute_similarity(documents)
    similarity_score = similarity_matrix[0, 1] * 100  # Convert to percentage

    result_text = f"Similarity between {filenames[0]} and {filenames[1]}: {similarity_score:.2f}%"
    result_label.config(text=result_text)

# Create the main GUI window
root = tk.Tk()
root.title("Plagiarism Detection System")
root.geometry("600x500")

# Add GUI components
title_label = tk.Label(root, text="Plagiarism Detection System", font=("Arial", 16))
title_label.pack(pady=10)

select_button = tk.Button(root, text="Select Multiple Documents", command=analyze_documents, font=("Arial", 12))
select_button.pack(pady=10)

compare_button = tk.Button(root, text="Compare Two Files", command=select_two_files_and_compare, font=("Arial", 12))
compare_button.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 12), justify="left", wraplength=500)
result_label.pack(pady=20)

# Run the GUI main loop
root.mainloop()
