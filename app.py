from flask import Flask, request, render_template
from configs import UNQ_CHARS
from utils import (
    CER_from_wavs,
    ctc_softmax_output_from_wavs,
    load_model,
    load_wav,
    plot_losses,
    predict_from_wavs,
)

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def translate_sound():
    if request.method == "POST":
        # Get the sound file from the request
        wavs = request.files["sound_file"]

        # Process the sound file using your AI model
        model = load_model("trained_model.h5")
        sentences, char_indices = predict_from_wavs(model, wavs, UNQ_CHARS)

        # Return the translation as a response
        return sentences

    # Render the webpage with the file upload form
    return render_template("index.html")


if __name__ == "__main__":
    app.run()
