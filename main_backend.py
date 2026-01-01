import os
import json
import io
import httpx
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from docx import Document
from typing import Literal, List, Optional, Dict, Any

# --- Configuration and Setup ---

# Use the official google-genai SDK for robustness and structured JSON output
try:
    from google import genai
    from google.genai.errors import APIError
except ImportError:
    # Fallback to allow setup, but raise error on execution if missing
    print("Warning: The 'google-genai' library is required. Please install it with 'pip install google-genai pydantic python-docx httpx'")

# Ensure the API key is available
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY environment variable not set. Please set it to your Gemini API key.")

# Initialize the Gemini Client
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    raise RuntimeError(f"Failed to initialize Gemini Client: {e}")

# Define the model and base URL
GEMINI_MODEL = "gemini-2.5-flash"

# --- Pydantic Models for Data Validation ---

class DomainTerm(BaseModel):
    """Model for an extracted legal domain term."""
    term: str = Field(..., description="The original legal term.")
    translation: str = Field(..., description="The translated term.")
    definition: str = Field(..., description="The plain language definition of the term.")

class GeminiResponseSchema(BaseModel):
    """The structured JSON response expected from the Gemini model."""
    translated_text: str = Field(..., description="The translated text with legal terms wrapped in <span> tags.")
    summary: str = Field(..., description="The role-adaptive summary of the translated text.")
    domain_terms: List[DomainTerm] = Field(..., description="List of all extracted legal domain terms.")

class TranslateResponse(BaseModel):
    """The final response structure for the API endpoint."""
    status: Literal["success", "error"]
    translated_text: str
    summary: str
    domain_terms: List[DomainTerm]

# --- Helper Functions ---

def extract_text_from_docx(file: UploadFile) -> str:
    """Extracts text content from a .docx file."""
    try:
        # Read the file content into an in-memory stream
        docx_content = io.BytesIO(file.file.read())
        document = Document(docx_content)
        text = "\n".join([paragraph.text for paragraph in document.paragraphs])
        if not text.strip():
            raise ValueError("The .docx file contains no extractable text.")
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing .docx file: {e}")

def create_system_prompt(source_lang: str, target_lang: str, user_role: str, include_summary: bool) -> str:
    """Generates the structured system prompt for Gemini."""
    summary_instructions = ""
    if include_summary:
        summary_instructions = f"""
4. If summarization is requested:
   - For user_role = "lawyer": provide a detailed summary preserving all legal terms and precision.
   - For user_role = "paralegal": provide a moderately simplified summary explaining legal terms.
   - For user_role = "common_man": provide a plain-language summary that simplifies legal content but keeps all legal terms explained clearly.
5. Ensure every legal term found in the source text appears meaningfully in the summary.
"""
    
    return f"""
You are a professional bilingual translator and legal language expert.

Perform the following steps:
1. Translate the given text from {source_lang} to {target_lang}.
2. Identify all legal domain-specific terms, listing each with its translation and definition.
3. Wrap every identified legal term in the translated text with:
   <span class="term" data-meaning="Definition of term">Term</span>
{summary_instructions}
Return the result as structured JSON:
{{
  "translated_text": "...",
  "summary": "...",
  "domain_terms": [
     {{ "term": "...", "translation": "...", "definition": "..." }}
  ]
}}
"""

# --- FastAPI App Initialization ---

app = FastAPI(
    title="TARSLATE Backend",
    description="Legal-domain translation and role-adaptive summarization platform using Gemini 2.5 Flash.",
    version="1.0.0"
)

# CORS Middleware
origins = ["*"]  # Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Endpoints ---

@app.get("/api/status", response_model=Dict[str, str])
async def get_status():
    """Endpoint 2: Check the backend status."""
    return {"status": "Backend running"}

@app.post("/api/translate", response_model=TranslateResponse)
async def translate_and_summarize(
    # File upload is optional, depends on the 'text' field being empty
    file: Optional[UploadFile] = None, 
    # Raw text input, can be empty if a file is uploaded
    text: Optional[str] = Form(None), 
    source_language: Literal["english", "tamil"] = Form(...),
    target_language: Literal["english", "tamil"] = Form(...),
    include_summary: bool = Form(False),
    user_role: Literal["lawyer", "paralegal", "common_man"] = Form("common_man"),
):
    """
    Endpoint 1: Translates text/document and provides role-adaptive summarization.
    Accepts raw text or a .docx file.
    """
    
    # 1. Determine input text source
    input_text = ""
    if file and file.filename:
        # Check file extension
        if file.filename.endswith('.docx'):
            input_text = extract_text_from_docx(file)
        else:
            raise HTTPException(status_code=400, detail="Only .docx file uploads are supported.")
    elif text:
        input_text = text
    
    if not input_text:
        raise HTTPException(status_code=400, detail="Must provide either 'text' or a '.docx' 'file' for translation.")

    # 2. Build the AI prompt
    system_prompt = create_system_prompt(
        source_language, 
        target_language, 
        user_role, 
        include_summary
    )
    
    # 3. Call the Gemini API
    try:
        # Simplified config - removing the problematic timeout field
        config = genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=GeminiResponseSchema, # Pass the pydantic model
        )
        
        # The prompt is the input text itself
        # Note: We move the timeout to the generate_content call if needed, 
        # or rely on the default for now to ensure stability.
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[input_text],
            config=config,
        )

        # 4. Parse and Validate the Response
        # The response.text is a JSON string conforming to the schema
        # We parse it and validate it against the pydantic model
        if not response.text:
             # This should not happen if response_mime_type is set, but as a safeguard
             raise APIError("Gemini returned an empty response text.")

        raw_json_data = json.loads(response.text)
        validated_data = GeminiResponseSchema(**raw_json_data)
        
        # 5. Return the structured response
        return TranslateResponse(
            status="success",
            translated_text=validated_data.translated_text,
            # Ensure summary is empty string if not requested, though Gemini should handle it
            summary=validated_data.summary if include_summary else "Summary not requested.",
            domain_terms=validated_data.domain_terms
        )

    except APIError as e:
        print(f"Gemini API Error: {e}")
        raise HTTPException(status_code=500, detail=f"AI service error: {e}")
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"JSON/Pydantic Validation Error: {e}")
        # Log the raw text to debug the model's output formatting
        error_detail = f"AI output format error. Could not parse response into expected structure. Detail: {e}"
        # If response.text is available, you could include it for debugging
        # error_detail += f" Raw output: {getattr(response, 'text', 'N/A')}"
        raise HTTPException(status_code=500, detail=error_detail)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {e}")

# To run this file:
# 1. Set the environment variable: export GEMINI_API_KEY="YOUR_API_KEY"
# 2. Run the application: uvicorn main_backend:app --reload