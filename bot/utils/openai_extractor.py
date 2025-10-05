"""
Estrattore PDF usando OpenAI API - Leggero per Raspberry Pi
Usa l'API di OpenAI invece di modelli locali pesanti
"""
import fitz  # PyMuPDF
import openai
import os
import re
from typing import List, Dict
import time

class OpenAILightExtractor:
    """
    Usa OpenAI API per convertire PDF in markdown strutturato
    Vantaggi:
    - Zero modelli locali (solo API calls)
    - Molto leggero per Raspberry Pi
    - Alta qualità di conversione
    - Supporta GPT-4o-mini (economico)
    """
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model  # gpt-4o-mini è economico e perfetto per questo task
        self.client = openai.OpenAI(api_key=self.api_key)
        self.max_tokens_per_request = 3000  # Limite per evitare costi eccessivi
        
    def chunk_text_smart(self, text: str) -> List[str]:
        """Divide il testo in modo intelligente per non superare i token limit"""
        # Stima circa 4 caratteri per token
        max_chars = self.max_tokens_per_request * 4
        
        # Prima prova a dividere per pagine
        pages = text.split('--- Page')
        chunks = []
        current_chunk = ""
        
        for page in pages:
            if len(current_chunk + page) < max_chars:
                current_chunk += "--- Page" + page if page != pages[0] else page
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = "--- Page" + page if page != pages[0] else page
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def extract_markdown_with_openai(self, text_chunk: str) -> str:
        """Converte un chunk di testo in markdown usando OpenAI"""
        
        system_prompt = """You are an expert at converting game manual text to well-structured markdown.

Your task:
1. Identify the hierarchical structure of the text
2. Convert to proper markdown with headers
3. Preserve all important information
4. Use consistent header levels

CRITICAL RULES:
- ONLY use information that appears in the provided text
- NEVER add external knowledge about the game
- NEVER add explanations, clarifications, or additional context
- NEVER infer information that is not explicitly stated
- NEVER add content from your knowledge of similar games
- ALWAYS preserve page references (--- Page X ---) exactly as they appear
- Keep page markers visible in the final markdown for indexing purposes

Formatting Rules:
- Use # for main sections (SETUP, GAMEPLAY, COMPONENTS, RULES, etc.)
- Use ## for subsections (Game Overview, Turn Sequence, etc.)  
- Use ### for sub-subsections and specific rules
- Keep numbered and bulleted lists in markdown format
- Preserve important formatting like bold text
- Keep page references intact: --- Page X --- should remain visible
- Don't remove page numbers or page markers
- Don't add content that wasn't in the original text

Focus on creating a clean, structured markdown that maintains the logical flow of the game manual using ONLY the information provided in the input text, while preserving all page references for vector database indexing."""

        user_prompt = f"""Convert this game manual text to markdown using ONLY the information provided below. Do not add any external knowledge or explanations.

IMPORTANT: Keep all page references (--- Page X ---) exactly as they appear - these are needed for vector database indexing.

Original manual text:
{text_chunk}

Convert to markdown format using ONLY the content above, preserving all page references. Return ONLY the markdown formatted text, no explanations or additions."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,
                temperature=0.1,  # Bassa per consistenza
                top_p=0.9
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Errore OpenAI API: {e}")
            # Se l'API fallisce, restituisce il testo originale
            return text_chunk
    
    def extract_from_pdf(self, pdf_path: str, progress_callback=None) -> str:
        """Estrae markdown da PDF usando OpenAI API"""
        if not self.api_key:
            raise ValueError("OpenAI API key non trovata. Imposta OPENAI_API_KEY environment variable.")
        
        # Estrai testo con PyMuPDF
        print("📄 Estraendo testo dal PDF...")
        doc = fitz.open(pdf_path)
        full_text = ""
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():  # Solo pagine con contenuto
                full_text += f"\n--- Page {page_num + 1} ---\n{text}"
        
        doc.close()
        
        if not full_text.strip():
            raise ValueError("Nessun testo estratto dal PDF")
        
        # Dividi in chunks
        print("🔄 Dividendo il testo in chunks...")
        chunks = self.chunk_text_smart(full_text)
        
        # Processa ogni chunk con OpenAI
        markdown_parts = []
        total_chunks = len(chunks)
        
        for i, chunk in enumerate(chunks):
            print(f"🤖 Processando chunk {i+1}/{total_chunks} con OpenAI...")
            
            # Callback per progress
            if progress_callback:
                progress_callback(i + 1, total_chunks)
            
            markdown_chunk = self.extract_markdown_with_openai(chunk)
            markdown_parts.append(markdown_chunk)
            
            # Piccola pausa per evitare rate limiting
            if i < total_chunks - 1:
                time.sleep(0.5)
        
        print("✅ Conversione completata!")
        return '\n\n'.join(markdown_parts)
    
    def extract_from_pdf_with_page_mapping(self, pdf_path: str, progress_callback=None) -> List[Dict]:
        """
        Estrae markdown da PDF mantenendo mappatura esplicita pagina-contenuto per Qdrant
        Restituisce una lista di dict con {content, page_start, page_end, markdown}
        """
        if not self.api_key:
            raise ValueError("OpenAI API key non trovata. Imposta OPENAI_API_KEY environment variable.")
        
        print("📄 Estraendo testo dal PDF con mappatura pagine...")
        doc = fitz.open(pdf_path)
        
        # Estrai contenuto per singola pagina
        page_contents = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                page_contents.append({
                    'page_num': page_num + 1,
                    'raw_text': text.strip(),
                    'markdown': None
                })
        
        doc.close()
        
        if not page_contents:
            raise ValueError("Nessun testo estratto dal PDF")
        
        # Processa ogni pagina con OpenAI mantenendo riferimenti
        processed_pages = []
        total_pages = len(page_contents)
        
        for i, page_data in enumerate(page_contents):
            print(f"🤖 Processando pagina {page_data['page_num']}/{total_pages} con OpenAI...")
            
            if progress_callback:
                progress_callback(i + 1, total_pages)
            
            # Aggiungi esplicitamente il numero di pagina al testo
            text_with_page = f"--- Page {page_data['page_num']} ---\n{page_data['raw_text']}"
            
            # Converti in markdown mantenendo riferimento pagina
            markdown_content = self.extract_markdown_with_openai(text_with_page)
            
            processed_pages.append({
                'content': page_data['raw_text'],
                'markdown': markdown_content,
                'page_start': page_data['page_num'],
                'page_end': page_data['page_num'],
                'page_reference': f"Page {page_data['page_num']}"
            })
            
            # Piccola pausa per evitare rate limiting
            if i < total_pages - 1:
                time.sleep(0.5)
        
        print("✅ Conversione con mappatura pagine completata!")
        return processed_pages
    
    def estimate_cost(self, pdf_path: str) -> Dict[str, float]:
        """Stima il costo dell'elaborazione"""
        doc = fitz.open(pdf_path)
        total_chars = 0
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            total_chars += len(text)
        
        doc.close()
        
        # Stima token (circa 4 caratteri per token)
        estimated_tokens = total_chars / 4
        
        # Prezzi GPT-4o-mini (Ottobre 2024)
        input_cost_per_1k = 0.00015  # $0.15 per 1K input tokens
        output_cost_per_1k = 0.0006  # $0.60 per 1K output tokens
        
        # Stima output tokens (circa 50% degli input per la conversione)
        output_tokens = estimated_tokens * 0.5
        
        input_cost = (estimated_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k
        total_cost = input_cost + output_cost
        
        return {
            "estimated_input_tokens": int(estimated_tokens),
            "estimated_output_tokens": int(output_tokens),
            "estimated_input_cost_usd": round(input_cost, 4),
            "estimated_output_cost_usd": round(output_cost, 4),
            "estimated_total_cost_usd": round(total_cost, 4)
        }


# Funzione helper per sostituire facilmente Marker
def create_openai_converter(api_key: str = None):
    """Crea un converter compatibile con l'interfaccia di Marker"""
    extractor = OpenAILightExtractor(api_key=api_key)
    
    def converter_func(pdf_path: str):
        """Funzione che simula l'interfaccia di Marker"""
        return extractor.extract_from_pdf(pdf_path)
    
    return converter_func

def openai_text_from_rendered(markdown_text):
    """Sostituisce text_from_rendered di Marker"""
    return markdown_text, None, None


if __name__ == "__main__":
    # Test
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Imposta OPENAI_API_KEY environment variable")
        exit(1)
    
    extractor = OpenAILightExtractor(api_key=api_key)
    
    # Test con testo di esempio
    test_text = """1. SETUP
Place the game board in the center of the table.

COMPONENTS
The game includes:
- 1 game board
- 52 playing cards
- 4 player tokens

1.1 Initial Setup
Each player starts with 7 cards in their hand.

GAMEPLAY
Players take turns in clockwise order."""
    
    print("🧪 Test conversione con OpenAI...")
    try:
        result = extractor.extract_markdown_with_openai(test_text)
        print("✅ Test riuscito!")
        print("Risultato:", result[:300] + "..." if len(result) > 300 else result)
    except Exception as e:
        print(f"❌ Errore test: {e}")
        print("Il test richiede una API key valida per funzionare.")