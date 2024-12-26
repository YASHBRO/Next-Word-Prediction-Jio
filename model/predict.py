import joblib
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences


class ModelPredictor:
    def __init__(self, model_path: str, tokenizer_path: str):
        self.model = load_model(model_path)
        self.tokenizer = joblib.load(tokenizer_path)

    def predict(self, text: str):
        token_list = self.tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], padding="pre")
        predicted = self.model.predict(token_list, verbose=0)
        return {
            "text": text,
            "prediction": self.tokenizer.index_word[predicted.argmax()],
        }


if __name__ == "__main__":
    predictor = ModelPredictor(
        "model/text_generation_model.keras",
        "model/tokenizer.pkl",
    )
    print(predictor.predict("Machine learning"))
