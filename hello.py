import PyPDF2
import ollama


def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text


def initialize_system_prompt(pdf_content):
    return (
            "You are a helpful assistant with the ability to answer questions based on the following document:\n\n"
            + pdf_content
            + "\n\nFeel free to answer questions or provide clarifications based on the document."
    )


def stream_response(messages):
    response = ollama.chat(messages=messages, model="llama3.2", stream=True)
    response_text = ""
    for chunk in response:
        if hasattr(chunk, "message") and hasattr(chunk.message, "content"):
            print(chunk.message.content, end="")
            response_text += chunk.message.content
    print("\n")
    return response_text


# Chatbot function
def chatbot_with_file_context(file_path):
    pdf_content = extract_text_from_pdf(file_path)

    system_prompt = initialize_system_prompt(pdf_content)

    print("Chatbot: Hi there! Ask me anything about the document.")
    messages = [{"role": "system", "content": system_prompt}]

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit", "bye"}:
            print("Chatbot: Goodbye! Have a great day!")
            break

        messages.append({"role": "user", "content": user_input})

        response_content = stream_response(messages)
        messages.append({"role": "assistant", "content": response_content})


if __name__ == "__main__":
    pdf_path = "Policies.pdf"
    chatbot_with_file_context(pdf_path)
