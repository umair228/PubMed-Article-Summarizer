# PubMed Article Summarization Web Application

## Overview
PubMed Article Summarization Web Application is a Flask-based web application that facilitates summarization of PubMed articles using natural language processing techniques. Users can input PubMed article text and obtain concise summaries either through generative AI models or rule-based methods.

## Features
- **Input Interface**: Users can enter PubMed article text into a text area.
- **Summarization Options**:
  - **Generative AI Model**: Utilizes the T5 model from Hugging Face for advanced text summarization.
  - **Rule-based Summarization**: Available as an alternative method.
- **Display**: Shows both the original article and the generated summary side by side.
- **Summary Length Customization**: Users can specify the desired length of the summary using a slider.
- **Copy to Clipboard**: Enables users to copy the generated summary text with a single click.
- **Error Handling**: Validates input to ensure valid PubMed article text is provided.

## Requirements
- Python 3.6+
- Flask
- requests
- transformers (Hugging Face library)
- torch

## Installation
1. Clone the repository:
```bash
git clone https://github.com/umair228/PubMed-Article-Summarizer.git
cd PubMed-Article-Summarizer
```

3. Install dependencies:
```python
pip install -r requirements.txt
```

## Usage
1. Run the Flask application:
```python
python app.py
```
3. Open a web browser and go to `http://localhost:5000` to access the application.
4. Enter PubMed article text, select summarization options, and click "Submit".
5. The summarized text will be displayed in the adjacent text area. Use "Copy text" button to copy the summary to clipboard.

## Optional Enhancements (Advanced Features)
- **User Authentication**: Implement user accounts to save and manage summaries.
- **Summary Options**: Add features to choose summary length or style (e.g., brief, detailed).

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
