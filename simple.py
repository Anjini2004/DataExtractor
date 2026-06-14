from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from pypdf import PdfReader
from bs4 import BeautifulSoup
import requests
import json
import os


# Create memory file if it doesn't exist
if not os.path.exists("memory.json"):
    with open("memory.json", "w") as f:
        json.dump({}, f)

@tool
def save_json(filename: str, data: str) -> str:
    """Save valid JSON data to a file."""

    if not filename.endswith(".json"):
        filename += ".json"

    try:
        parsed_data = json.loads(data)
    except json.JSONDecodeError:
        return "Error: data was not valid JSON, so it was not saved."

    with open(filename, "w") as f:
        json.dump(parsed_data, f, indent=4)

    return f"Saved valid JSON to {filename}"

def extract_json(text: str):
    """Extract JSON object from model output."""

    text = text.strip()

    # Remove markdown fences if present
    text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object found")

    json_text = text[start:end + 1]

    return json.loads(json_text)

@tool
def save_memory(key: str, value: str) -> str:
    """Save information about the user."""

    with open("memory.json", "r") as f:
        memory = json.load(f)

    memory[key] = value

    with open("memory.json", "w") as f:
        json.dump(memory, f, indent=4)

    return f"Saved {key}: {value}"


@tool
def get_memory(key: str) -> str:
    """Retrieve stored information."""

    with open("memory.json", "r") as f:
        memory = json.load(f)

    return memory.get(key, "Not found")


@tool
def get_resume_data(path: str) -> str:
    """Read text from a PDF resume file."""

    if not os.path.exists(path):
        return f"File not found: {path}. Available files: {os.listdir()}"

    if os.path.getsize(path) == 0:
        return f"File exists but is empty: {path}"

    try:
        reader = PdfReader(path)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        if not text.strip():
            return "No text could be extracted from the PDF. It may be scanned/image-based."

        return text[:8000]

    except Exception as e:
        return f"Could not read PDF: {str(e)}"
    
@tool
def list_files() -> str:
    """List all files available in the current working directory. Use this tool whenever the user asks what files exist, what files are available, or asks to list files."""

    files = []

    for file in os.listdir():
        if os.path.isfile(file):
            files.append(f"{file} - {os.path.getsize(file)} bytes")

    return "\n".join(files)

# Local Ollama model
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


agent = create_agent(
    model=llm,
    tools=[
        save_memory,
        get_memory,
        get_resume_data,
        save_json,
        list_files
    ],
    system_prompt="""
You are a helpful assistant.

If the user provides a PDF filename and asks you to extract information:
1. Use get_resume_data to read the PDF.
2. Return ONLY valid JSON.
3. Do not use markdown.
4. Do not include explanations.
5. Do not wrap the JSON in ```json.
6. The JSON must use key-value format.

For normal conversation, answer normally.
""")


while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        break

    # If user gives a PDF filename, handle it reliably with Python
    if user_input.lower().endswith(".pdf"):
        resume_text = get_resume_data.invoke({"path": user_input})

        extraction_prompt = f"""
Extract structured data from the following PDF text.

Return ONLY valid JSON.
Do not include markdown.
Do not include explanation.

PDF text:
{resume_text}
"""

        response = llm.invoke(extraction_prompt)
        content = response.content

        print("\nAgent:", content)

        try:
            parsed = extract_json(content)

            with open("data.json", "w") as f:
                json.dump(parsed, f, indent=4)

            print("\nSaved to data.json")

        except ValueError as e:
            print(f"\nCould not save because the model did not return valid JSON: {e}")

        continue

    # Normal chatbot behavior
    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": user_input
            }
        ]
    })

    final_message = result["messages"][-1]
    content = final_message.content

    if isinstance(content, list):
        content = content[0]["text"]

    print("\nAgent:", content)