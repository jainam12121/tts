import onnxruntime
import soundfile as sf
import yaml

from ttstokenizer import TTSTokenizer

# Load configuration
with open("./config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Initialize ONNX model
model = onnxruntime.InferenceSession(
    "./model.onnx",
    providers=["CPUExecutionProvider"]
)

# Create tokenizer
tokenizer = TTSTokenizer(config["token"]["list"])

def pre_process(text):
    """Tokenizes input text."""
    print("In Custom pre_process method")
    print("Input text:", text)
    tokenized_input = tokenizer(text)
    print("Tokenized input:", tokenized_input)
    return tokenized_input

def post_process(output):
    """Processes model output and saves it as a .wav file."""
    print("In Custom post_process method")
    audio_data = output[0]
    output_file = "out.wav"
    sf.write(output_file, audio_data, 22050)
    print("Audio written to:", output_file)
    return output_file

# Example input
input_text = "Say something here"

# Pre-process the input
processed_input = pre_process(input_text)

# Run the model with the pre-processed input
outputs = model.run(None, {"text": processed_input})

# Post-process the output
output_file = post_process(outputs)