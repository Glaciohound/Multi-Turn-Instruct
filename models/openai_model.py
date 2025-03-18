import openai
from .model_base import ModelBase


class OpenAIModel(ModelBase):

    model_name_list = [
        "o1-preview", "o1-mini",
        "gpt-4o", "gpt-4", "gpt-4o-mini", "gpt-4-turbo",
        "gpt-3.5-turbo", "gpt-3.5-turbo-instruct", "gpt-3.5-turbo-0125"
    ]

    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=api_key)

    def respond(self, messages, max_tokens, max_context_size):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_completion_tokens=max_tokens
            )
            response = response.choices[0].message.content
        except openai.BadRequestError as e:
            response = e.body["message"]
        return response
