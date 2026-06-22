# VeriGraph AI

VeriGraph AI is a Python-based intelligent knowledge graph builder and validation system.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd VeriGraph_AI
   ```

2. **Set up the virtual environment:**
   ```bash
   python -m venv verigraph
   .\verigraph\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Create a `.env` file in the root directory and add your API keys:
   ```env
   OPENAI_API_KEY=your_openai_key
   GOOGLE_API_KEY=your_google_key
   GROQ_API_KEY=your_groq_key
   ```

5. **Run the application:**
   ```bash
   python -m app.main
   ```
