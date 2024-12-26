import tensorflow as tf
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Embedding, LSTM, Dense
import joblib


def train(data_path: str, model_path: str, tokenizer_path: str):
    # Read the text file and create corpus
    with open(data_path, "r", encoding="utf-8") as file:
        corpus = file.read().splitlines()

    # Tokenize the corpus
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    # Create input sequences
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[: i + 1]
            input_sequences.append(n_gram_sequence)

    # Pad sequences
    input_sequences = pad_sequences(input_sequences, padding="pre")

    # Create predictors and labels
    X, y = input_sequences[:, :-1], input_sequences[:, -1]
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)

    # Define the model
    model = Sequential()
    model.add(
        Embedding(
            total_words,
            10,
        )
    )
    model.add(LSTM(100))
    model.add(Dense(total_words, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    # Train the model
    model.fit(X, y, epochs=10, verbose=1)

    model.save(model_path)
    joblib.dump(tokenizer, tokenizer_path)

    print(f"Model saved at {model_path}")


if __name__ == "__main__":
    train(
        "dataset.txt",
        "model/text_generation_model.keras",
        "model/tokenizer.pkl",
    )
