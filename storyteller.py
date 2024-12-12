from utils.text_utils import extract_choices_and_intro
from utils.text_generation import generate_text

def respond_storyteller(
    message,
    system_message,
    max_tokens,
    temperature,
    top_p,
    shared_history
):

    messages = [{"role": "system", "content": str(system_message)}]

    for val in shared_history[-3:]:
        if val["role"] and val["content"][:14] == "storyteller: ":
            messages.append({"role": val["role"], "content": val["content"]})
        elif val["role"] and val == shared_history[-1] and val["content"][:13] == "strong agent: ":
            messages.append({"role": val["role"], "content": val["content"]})

    # Concatenate message and previous history to generate input for GPT-4o mini
    prompt = ' '.join([m["content"] for m in messages]) + ' ' + message

    # Use GPT-4o mini to generate text
    response = generate_text(prompt, system_message, max_length=max_tokens, temperature=temperature, top_p=top_p)

    yield response
