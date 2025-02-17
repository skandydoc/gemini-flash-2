# Gemini Flash Chat Interface

A powerful chat interface for processing multiple files using Google's Gemini Flash 2.0 model. This application allows users to upload multiple files and process them with custom prompts, displaying results in various formats including tables, text, code, and charts.

Composed with Cursor - Use with Discretion. AI hallucinations and potential errors are possible.

## Features

- Multiple file upload support
- Custom prompt input for file processing
- Dynamic response visualization
- Support for various output formats (tables, text, code, charts)
- Error handling and graceful degradation
- Progress tracking for file processing

## Setup

1. Clone the repository:
```bash
git clone git@github.com:skandydoc/gemini-flash-2.git
cd gemini-flash-2
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

5. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Upload one or more files using the file upload interface
2. Enter your custom prompt in the text area
3. Click "Process Files" to start processing
4. View results in the dynamic output display

## License

This project is licensed under CC BY-SA NC 4.0 (Creative Commons Attribution-ShareAlike Non-commercial 4.0 International licence).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 