import os
import openai

openai.api_key = os.getenv("")

response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages= [
                    {'role': 'user', 'content': 'Translate the following English text to French: '}
                ]
            )
