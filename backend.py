from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import random
import json
import pandas as pd

app = FastAPI()

model = tf.keras.models.load_model(r'D:\AI-ML\NLP\ImperfectBastards\Chatbot\chattybot.h5')

with open('tokenizer.json') as f:
    tokenizer_data = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_data)

with open('intents.json') as f:
    intents = json.load(f)

tags = [intent['tag'] for intent in intents['intents']]

y = pd.Series(tags).astype('category')

label_map = dict(enumerate(y.cat.categories))

reverse_label_map = {v: k for k, v in label_map.items()}

responses = {intent['tag']: intent['responses'] for intent in intents['intents']}

class ModelInput(BaseModel):
    text: str

def get_response(tag):
    return random.choice(responses.get(tag, ["I don't understand that."]))

@app.post("/predict/")
def predict(input: ModelInput):
    try:
        sequences = tokenizer.texts_to_sequences([input.text])
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=9)
        predictions = model.predict(padded_sequences)
        predicted_label = np.argmax(predictions, axis=1)[0]
        predicted_tag = label_map[predicted_label]
        response = get_response(predicted_tag)
        
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
