import google.generativeai as genai

def create_gpt_request(prompt, count):
    req = {}

    messages = []

    message1 = {}
    message2 = {}
    message1["role"] = "system"
    message1["content"] = "You are a helpful assistant."
    message2["role"] = "user"
    message2["content"] = prompt

    messages.append(message1)
    messages.append(message2)

    body = {}
    model = "gpt-4o-mini"
    body["model"] = model
    body["messages"] = messages
    body["max_tokens"] = 500

    id = "request-" + str(count)

    req["custom_id"] = id
    req["method"] = "POST"
    req["url"] = "/v1/chat/completions"
    req["body"] = body

    return req


def get_gemini(google_api_key: str):
    genai.configure(api_key=google_api_key)

    generation_config = {
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
    }

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-latest",
        generation_config=generation_config,
        safety_settings=safety_settings,
    )
    return model
