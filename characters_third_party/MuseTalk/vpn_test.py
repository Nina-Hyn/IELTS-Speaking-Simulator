import openai
import os

# 设置 API 密钥
openai.api_key = "your-api-key"

# 设置代理环境变量
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

try:
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Tell me a joke",
        max_tokens=50
    )
    print(response.choices[0].text)
except openai.error.AuthenticationError as e:
    print(f"Authentication error: {e}")
except openai.error.APIConnectionError as e:
    print(f"API connection error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")