from transformers import pipeline
from guidance import models, gen

device = "mps"

# neural_chat = models.TransformersChat(model="Intel/neural-chat-7b-v3-1", device=device)
neural_chat = models.TransformersChat(model="mistralai/Mistral-7B-v0.1", device=device)
# neural_chat = models.TransformersChat(model="meta-llama/Llama-2-13b-hf", device=device)
answer = neural_chat + f'Do you want a joke or a poem? ' + gen(stop='.')
print(f"Answer: {answer}")

# Load a pre-trained model for sentiment analysis
sentiment_pipeline = pipeline("sentiment-analysis", device=device)
text = "I love using transformers! They make machine learning much easier."
result = sentiment_pipeline(text)
print(f"Result: {result}")