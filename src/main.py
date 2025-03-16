import os
import json
import pickle
import time
import sys

# Set the project root (parent directory of src/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add project root to sys.path so that config.py (located in the project root) can be imported.
sys.path.insert(0, project_root)

import config
import openai
from sentence_transformers import SentenceTransformer

openai.api_key = config.OPENAI_API_KEY

client = openai
import numpy as np
import faiss

def read_settings(file_name):
    settings = {}
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            settings[key.strip()] = value.strip()
    return settings

# Use the project root instead of the src/ folder.
settings_path = os.path.join(project_root, "settings.txt")
settings = read_settings(settings_path)

# Set the OpenAI API key from config.py.
os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

# Global variable to store context from a previous session.
last_session = None

def load_faiss_resources():
    # Use the project root to locate the data folder.
    data_dir = os.path.join(project_root, "data")
    faiss_index_path = os.path.join(data_dir, "faiss_index.bin")
    metadata_path = os.path.join(data_dir, "faiss_metadata.json")
    if not os.path.exists(faiss_index_path) or not os.path.exists(metadata_path):
        print("FAISS index or metadata not found. Please run the load processed data script first.")
        sys.exit(1)
    index = faiss.read_index(faiss_index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

faiss_index, faiss_metadata = load_faiss_resources()

sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_query(query):
    embedding = sentence_model.encode(query)
    return np.array(embedding, dtype=np.float32)

def get_context_from_query(query, k=3):
    query_embedding = embed_query(query)
    query_embedding = np.expand_dims(query_embedding, axis=0)  # Shape (1, dim)
    distances, indices = faiss_index.search(query_embedding, k)
    context_chunks = []
    for idx in indices[0]:
        if idx < len(faiss_metadata):
            context_chunks.append(faiss_metadata[idx]["chunk_text"])
    context = "\n\n".join(context_chunks)
    return context

def verify_answer(original_question, answer):
    verification_prompt = [
        {"role": "system", "content": "Just say 'Yes' or 'No'. Do not give any other answer."},
        {"role": "user", "content": f"User: {original_question}\nAttendant: {answer}\nWas the Attendant able to answer the user's question?"}
    ]
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=5,
        temperature=0.0,
        messages=verification_prompt
    )
    verification = response.choices[0].message.content.strip().lower()
    return verification.startswith("y")

def check_syllabus(question, classname, professor, assistants, classdescription):
    prompt = [
        {"role": "user", "content": (
            f"This question is from a student in an {classname} taught by {professor} with the help of {assistants}. "
            f"The class is {classdescription}. I want to know whether this question is likely about logistical details, "
            f"schedule, nature, teachers, assignments, or the syllabus of the course? Answer Yes or No and nothing else: {question}"
        )}
    ]
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=5,
        temperature=0.0,
        messages=prompt
    )
    result = response.choices[0].message.content.strip().lower()
    return result.startswith("y")

def check_followup(new_question, previous_context):
    prompt = [
        {"role": "user", "content": (
            f"Consider this new question: {new_question}. The previous question and response was: {previous_context}. "
            "Would it be helpful to include the previous context to answer the new question? Answer Yes or No."
        )}
    ]
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=5,
        temperature=0.0,
        messages=prompt
    )
    result = response.choices[0].message.content.strip().lower()
    return result.startswith("y")

def main():
    global last_session

    # Course-specific details from settings.
    classname = settings.get("classname", "")
    professor = settings.get("professor", "")
    assistants = settings.get("assistants", "")
    classdescription = settings.get("classdescription", "")
    instructions = settings.get("instructions", "")
    assistant_name = settings.get("assistantname", "Virtual Assistant")

    # Prompt user input via terminal.
    user_input = input("Enter your prompt: ").strip()

    # Determine question type by prefix:
    # "m:" for multiple choice, "a:" for answer-check, otherwise normal.
    question_type = "normal"
    if user_input.lower().startswith("m:"):
        question_type = "multiple_choice"
        user_input = user_input[2:].strip()
    elif user_input.lower().startswith("a:"):
        question_type = "answer_check"
        user_input = user_input[2:].strip()

    original_question = user_input

    if question_type == "normal":
        if check_syllabus(user_input, classname, professor, assistants, classdescription):
            print("Detected syllabus-related question; modifying query accordingly.")
            original_question = f"I may be asking about a detail on the syllabus for {classname}. {user_input}"
        if last_session and check_followup(user_input, last_session):
            print("Detected follow-up question; incorporating previous context.")
            original_question = f"I have a follow-up on the previous question and response. {last_session} My new question is: {user_input}"

    # Retrieve context using FAISS (unless answer-check).
    context = ""
    if question_type != "answer_check":
        context = get_context_from_query(original_question, k=3)
        print("Retrieved relevant context from course materials.")
    else:
        if last_session:
            context = last_session
        else:
            print("No previous session context available for answer-check.")

    # Build system instructions based on question type.
    if question_type == "multiple_choice":
        prompt_instructions = (
            f"You are a very truthful, precise TA in a {classname}. You think step by step. A strong graduate student "
            f"is using you as a tutor. The student would like you to prepare a challenging multiple choice question on "
            "the requested topic drawing ONLY on the attached context. Do not refer to 'the attached context' explicitly. "
            "Present the question followed by options A to D. After the question, write <span style='display:none'> then give "
            "your answer and a short explanation, then close the span with </span>."
        )
        final_query = f"Construct a challenging multiple-choice question to test me on a concept related to {original_question}"
    elif question_type == "answer_check":
        prompt_instructions = (
            f"You are a very truthful, precise TA in a {classname}. You think step by step. You are testing a strong graduate student "
            "on their knowledge. Using the attached context, tell me whether the attached multiple choice answer is correct. "
            "Draw ONLY on the context for definitions and theoretical content. Do not refer to 'the attached context'. Just state "
            "your answer and rationale."
        )
        final_query = original_question
    else:
        prompt_instructions = (
            f"You are a very truthful, precise TA in a {classname}, a {classdescription}. You think step by step. A strong graduate "
            "student is asking you questions. Answer in no more than three paragraphs if the answer is found in the attached context. "
            "Do not restate the question or refer explicitly to the context. If you cannot find the answer in the context, say 'I don't know'."
        )
        final_query = original_question

    system_message = prompt_instructions + "\n\nContext:\n" + context

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": final_query}
    ]

    print("Sending query to GPT...")
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    reply = response.choices[0].message.content

    if question_type != "answer_check":
        last_session = context[:3900]

    if question_type != "multiple_choice":
        verified = verify_answer(original_question, reply)
        print("Answer verification:", "Yes" if verified else "No")
        if not verified and question_type != "answer_check":
            print("Attempting follow-up query with alternate context.")
            alternate_context = get_context_from_query(original_question + " " + context, k=5)
            followup_system = prompt_instructions + "\n\nContext:\n" + alternate_context
            followup_messages = [
                {"role": "system", "content": followup_system},
                {"role": "user", "content": final_query}
            ]
            followup_response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=followup_messages
            )
            followup_reply = followup_response.choices[0].message.content
            if verify_answer(original_question, followup_reply):
                reply = followup_reply
            else:
                reply = "I'm sorry but I cannot answer that question. Can you rephrase or ask an alternative?"

    print("\nFinal Answer:\n", reply)

if __name__ == "__main__":
    main()









