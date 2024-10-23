from flask import Flask, render_template, request, send_file
from src.sensor.pipeline.prediction import PredictionPipeline
import os

app = Flask(__name__)

# Helper function to convert Google Drive view link to direct download link
def convert_drive_url_to_direct_download(source_url):
    if 'drive.google.com' in source_url and 'view' in source_url:
        # Extract the file ID
        file_id = source_url.split('/d/')[1].split('/')[0]
        # Create a direct download URL
        direct_url = f"https://drive.google.com/uc?id={file_id}&export=download"
        return direct_url
    return source_url

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get the URL from the form input
        data_url = request.form["url"]

        # Convert the URL if it's a Google Drive view link
        data_url = convert_drive_url_to_direct_download(data_url)

        # Run the prediction pipeline using the converted URL
        pipeline = PredictionPipeline(data_url)
        try:
            output_csv = pipeline.run_pipeline()
            return send_file(output_csv, as_attachment=True)
        except Exception as e:
            return f"Error: {str(e)}"

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
