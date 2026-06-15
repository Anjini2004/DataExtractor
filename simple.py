from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from pypdf import PdfReader
from bs4 import BeautifulSoup
import requests
import json
import os
import re


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

def sanitize_json_text(text: str) -> str:
    """Escape invalid characters inside JSON string literals."""

    result = []
    in_string = False
    escape = False

    for ch in text:
        if escape:
            result.append(ch)
            escape = False
            continue

        if ch == "\\":
            result.append(ch)
            escape = True
            continue

        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue

        if in_string:
            if ch == "\n":
                result.append("\\n")
                continue
            if ch == "\r":
                result.append("\\r")
                continue
            if ch == "\t":
                result.append("\\t")
                continue
            if ord(ch) < 0x20:
                result.append(f"\\u{ord(ch):04x}")
                continue

        result.append(ch)

    return ''.join(result)


def extract_json(text: str):
    """Extract JSON object from model output."""

    text = text.strip()

    # Remove markdown fences if present
    text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found")

    depth = 0
    in_string = False
    escape = False
    end = None

    for i in range(start, len(text)):
        ch = text[i]

        if escape:
            escape = False
            continue

        if ch == "\\":
            escape = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end is None:
        raise ValueError("No complete JSON object found")

    json_text = text[start:end + 1]
    json_text = sanitize_json_text(json_text)

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        trailing_commas_fixed = re.sub(r',\s*(?=[}\]])', '', json_text)
        if trailing_commas_fixed != json_text:
            try:
                return json.loads(trailing_commas_fixed)
            except json.JSONDecodeError:
                pass

        snippet_start = max(0, e.pos - 40)
        snippet_end = min(len(json_text), e.pos + 40)
        snippet = json_text[snippet_start:snippet_end]
        raise ValueError(
            f"Invalid JSON: {e.msg} at position {e.pos}. "
            f"Snippet: {snippet!r}"
        ) from e

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
Return only the JSON object with no text before or after.

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