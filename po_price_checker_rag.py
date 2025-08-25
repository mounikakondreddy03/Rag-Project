import pandas as pd
import numpy as np
import PyPDF2
import io
import streamlit as st
from typing import List, Dict, Any, Tuple
import openai
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from dataclasses import dataclass
import re
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PriceDiscrepancy:
    item_code: str
    item_name: str
    po_price: float
    master_price: float
    difference: float
    percentage_diff: float
    status: str

class DocumentProcessor:
    """Handle PDF and Excel file processing"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return ""
    
    @staticmethod
    def parse_po_from_pdf(pdf_text: str) -> pd.DataFrame:
        """Parse PO data from PDF text using regex patterns"""
        try:
            item_patterns = [
                r'(\d+)\s+([A-Za-z0-9\s]+?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)',  
                r'([A-Za-z0-9]+)\s+(.+?)\s+\$?(\d+(?:\.\d+)?)',
            ]
            
            items = []
            for pattern in item_patterns:
                matches = re.findall(pattern, pdf_text)
                for match in matches:
                    if len(match) >= 3:
                        items.append({
                            'item_code': match[0].strip(),
                            'item_name': match[1].strip(),
                            'quantity': float(match[2]) if len(match) > 3 else 1,
                            'unit_price': float(match[-1])
                        })
            
            return pd.DataFrame(items)
        except Exception as e:
            logger.error(f"Error parsing PO PDF: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def load_excel_file(excel_file) -> pd.DataFrame:
        """Load Excel file and standardize column names"""
        try:
            df = pd.read_excel(excel_file)
            
            column_mapping = {
                'Item Code': 'item_code',
                'Item': 'item_code',
                'Code': 'item_code',
                'Item Name': 'item_name',
                'Name': 'item_name',
                'Description': 'item_name',
                'Price': 'unit_price',
                'Unit Price': 'unit_price',
                'Master Price': 'unit_price',
                'Cost': 'unit_price'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})
            
            return df
        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            return pd.DataFrame()

class VectorStore:
    """Handle vector storage and similarity search"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.metadata = []
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents to vector store"""
        try:
            embeddings = self.model.encode(documents)
            
            if self.index is None:
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatL2(dimension)
            
            self.index.add(embeddings.astype('float32'))
            self.documents.extend(documents)
            
            if metadata:
                self.metadata.extend(metadata)
            else:
                self.metadata.extend([{}] * len(documents))
                
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, Dict, float]]:
        """Search for similar documents"""
        try:
            if self.index is None:
                return []
            
            query_embedding = self.model.encode([query])
            distances, indices = self.index.search(query_embedding.astype('float32'), k)
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.documents):
                    results.append((
                        self.documents[idx],
                        self.metadata[idx],
                        float(distance)
                    ))
            
            return results
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []

class PriceAnalyzer:
    """Analyze price discrepancies between PO and Master Price List"""
    
    def __init__(self, tolerance_percentage: float = 5.0):
        self.tolerance_percentage = tolerance_percentage
    
    def compare_prices(self, po_df: pd.DataFrame, master_df: pd.DataFrame) -> List[PriceDiscrepancy]:
        """Compare prices between PO and master price list"""
        discrepancies = []
        
        try:
            # Merge dataframes on item_code
            merged_df = po_df.merge(
                master_df, 
                on='item_code', 
                how='left', 
                suffixes=('_po', '_master')
            )
            
            for _, row in merged_df.iterrows():
                if pd.notna(row.get('unit_price_master')):
                    po_price = float(row['unit_price_po'])
                    master_price = float(row['unit_price_master'])
                    
                    difference = po_price - master_price
                    percentage_diff = (difference / master_price) * 100 if master_price != 0 else 0
                    
                    # Determine status
                    if abs(percentage_diff) <= self.tolerance_percentage:
                        status = "Within Tolerance"
                    elif percentage_diff > 0:
                        status = "PO Price Higher"
                    else:
                        status = "PO Price Lower"
                    
                    discrepancies.append(PriceDiscrepancy(
                        item_code=row['item_code'],
                        item_name=row.get('item_name_po', row.get('item_name_master', '')),
                        po_price=po_price,
                        master_price=master_price,
                        difference=difference,
                        percentage_diff=percentage_diff,
                        status=status
                    ))
                else:
                    # Item not found in master price list
                    discrepancies.append(PriceDiscrepancy(
                        item_code=row['item_code'],
                        item_name=row.get('item_name_po', ''),
                        po_price=float(row['unit_price_po']),
                        master_price=0.0,
                        difference=0.0,
                        percentage_diff=0.0,
                        status="Not Found in Master List"
                    ))
            
            return discrepancies
        except Exception as e:
            logger.error(f"Error comparing prices: {e}")
            return []

class RAGSystem:
    """Main RAG system for price checking"""
    
    def __init__(self, openai_api_key: str = None):
        self.vector_store = VectorStore()
        self.document_processor = DocumentProcessor()
        self.price_analyzer = PriceAnalyzer()
        self.po_data = None
        self.master_data = None
        self.discrepancies = []
        
        if openai_api_key:
            openai.api_key = openai_api_key
    
    def process_po_file(self, uploaded_file) -> bool:
        """Process uploaded PO file"""
        try:
            if uploaded_file.name.endswith('.pdf'):
                pdf_text = self.document_processor.extract_text_from_pdf(uploaded_file)
                self.po_data = self.document_processor.parse_po_from_pdf(pdf_text)
                
                # Add PO text to vector store
                self.vector_store.add_documents(
                    [pdf_text],
                    [{"type": "po_document", "filename": uploaded_file.name}]
                )
            else:
                return False
            
            return len(self.po_data) > 0
        except Exception as e:
            logger.error(f"Error processing PO file: {e}")
            return False
    
    def process_master_price_file(self, uploaded_file) -> bool:
        """Process uploaded master price file"""
        try:
            if uploaded_file.name.endswith(('.xlsx', '.xls')):
                self.master_data = self.document_processor.load_excel_file(uploaded_file)
                
                # Create searchable documents from master price data
                documents = []
                metadata = []
                
                for _, row in self.master_data.iterrows():
                    doc = f"Item Code: {row.get('item_code', '')}, " \
                          f"Item Name: {row.get('item_name', '')}, " \
                          f"Price: {row.get('unit_price', '')}"
                    documents.append(doc)
                    metadata.append({
                        "type": "master_price_item",
                        "item_code": row.get('item_code', ''),
                        "item_name": row.get('item_name', ''),
                        "unit_price": row.get('unit_price', '')
                    })
                
                self.vector_store.add_documents(documents, metadata)
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"Error processing master price file: {e}")
            return False
    
    def analyze_prices(self) -> List[PriceDiscrepancy]:
        """Analyze price discrepancies"""
        if self.po_data is not None and self.master_data is not None:
            self.discrepancies = self.price_analyzer.compare_prices(self.po_data, self.master_data)
            return self.discrepancies
        return []
    
    def generate_response(self, query: str, use_openai: bool = False) -> str:
        """Generate response to user query using RAG"""
        try:
            # Search for relevant documents
            relevant_docs = self.vector_store.search(query, k=5)
            
            # Prepare context
            context = ""
            for doc, metadata, score in relevant_docs:
                context += f"Document: {doc}\n"
                if metadata:
                    context += f"Metadata: {metadata}\n"
                context += f"Relevance Score: {score}\n\n"
            
            # Add discrepancy information if available
            if self.discrepancies:
                context += "Price Analysis Results:\n"
                for disc in self.discrepancies[:10]:  # Limit to first 10
                    context += f"- {disc.item_code}: PO Price ${disc.po_price:.2f}, " \
                              f"Master Price ${disc.master_price:.2f}, Status: {disc.status}\n"
            
            if use_openai and openai.api_key:
                # Use OpenAI API
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant for price checking and PO analysis. Use the provided context to answer questions about purchase orders and price discrepancies."},
                        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
                    ],
                    max_tokens=500
                )
                return response.choices[0].message.content
            else:
                # Simple rule-based response
                return self._generate_simple_response(query, context)
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while processing your query."
    
    def _generate_simple_response(self, query: str, context: str) -> str:
        """Generate simple response without OpenAI"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['discrepancy', 'difference', 'mismatch']):
            if self.discrepancies:
                high_discrepancies = [d for d in self.discrepancies if abs(d.percentage_diff) > 10]
                if high_discrepancies:
                    response = f"Found {len(high_discrepancies)} items with significant price discrepancies (>10%):\n"
                    for disc in high_discrepancies[:5]:
                        response += f"- {disc.item_code}: {disc.percentage_diff:.1f}% difference\n"
                    return response
                else:
                    return "Most prices are within acceptable tolerance levels."
            else:
                return "Please upload both PO and Master Price files to analyze discrepancies."
        
        elif any(word in query_lower for word in ['summary', 'overview', 'analysis']):
            if self.discrepancies:
                total_items = len(self.discrepancies)
                within_tolerance = len([d for d in self.discrepancies if d.status == "Within Tolerance"])
                return f"Analysis Summary: {total_items} items analyzed, {within_tolerance} within tolerance, {total_items - within_tolerance} require attention."
            else:
                return "Please complete the price analysis first by uploading both files."
        
        else:
            return "I can help you analyze price discrepancies between your PO and Master Price List. Please ask about discrepancies, summary, or specific items."

# Streamlit UI
def main():
    st.set_page_config(page_title="PO ⟷ Master Price Checker (RAG)", layout="wide")
    
    st.title("PO ⟷ Master Price Checker (RAG)")
    st.write("Upload your PO and Master Price List to validate prices, analyze mismatches, and ask questions.")
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
    
    # Sidebar for file uploads
    st.sidebar.header("File Uploads")
    
    # PO Data Upload
    st.sidebar.subheader("Upload PO Data")
    po_file = st.sidebar.file_uploader(
        "Choose PO file",
        type=['pdf'],
        help="PDF files containing Purchase Order data"
    )
    
    if po_file is not None:
        if st.sidebar.button("Process PO File"):
            with st.spinner("Processing PO file..."):
                success = st.session_state.rag_system.process_po_file(po_file)
                if success:
                    st.sidebar.success("PO file processed successfully!")
                else:
                    st.sidebar.error("Error processing PO file")
    
    # Master Price List Upload
    st.sidebar.subheader("Upload Master Price List")
    master_file = st.sidebar.file_uploader(
        "Choose Master Price file",
        type=['xlsx', 'xls'],
        help="Excel files containing Master Price List"
    )
    
    if master_file is not None:
        if st.sidebar.button("Process Master Price File"):
            with st.spinner("Processing Master Price file..."):
                success = st.session_state.rag_system.process_master_price_file(master_file)
                if success:
                    st.sidebar.success("Master Price file processed successfully!")
                else:
                    st.sidebar.error("Error processing Master Price file")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Price Analysis")
        
        if st.button("Analyze Prices", type="primary"):
            if st.session_state.rag_system.po_data is not None and st.session_state.rag_system.master_data is not None:
                with st.spinner("Analyzing prices..."):
                    discrepancies = st.session_state.rag_system.analyze_prices()
                    
                    if discrepancies:
                        # Display results
                        st.subheader("Price Analysis Results")
                        
                        # Summary metrics
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Total Items", len(discrepancies))
                        with col_b:
                            within_tolerance = len([d for d in discrepancies if d.status == "Within Tolerance"])
                            st.metric("Within Tolerance", within_tolerance)
                        with col_c:
                            high_discrepancy = len([d for d in discrepancies if abs(d.percentage_diff) > 10])
                            st.metric("High Discrepancy (>10%)", high_discrepancy)
                        
                        # Detailed table
                        df_results = pd.DataFrame([
                            {
                                'Item Code': d.item_code,
                                'Item Name': d.item_name,
                                'PO Price': f"${d.po_price:.2f}",
                                'Master Price': f"${d.master_price:.2f}",
                                'Difference': f"${d.difference:.2f}",
                                'Percentage Diff': f"{d.percentage_diff:.1f}%",
                                'Status': d.status
                            }
                            for d in discrepancies
                        ])
                        
                        st.dataframe(df_results, use_container_width=True)
                    else:
                        st.warning("No discrepancies found or error in analysis")
            else:
                st.error("Please upload both PO and Master Price files first")
    
    with col2:
        st.header("Ask Questions")
        
        # Chat interface
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about price discrepancies..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.rag_system.generate_response(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()