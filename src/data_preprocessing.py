"""
Data preprocessing utilities for complaint analysis.

This module contains functions for loading, cleaning, and preprocessing
complaint data from the CFPB dataset.
"""

import pandas as pd
import numpy as np
import re
import os
from typing import List, Dict, Any, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings

warnings.filterwarnings('ignore')


class ComplaintDataProcessor:
    """
    Main class for processing complaint data from raw CSV to cleaned format.
    """
    
    def __init__(self):
        self.target_products = [
            'Credit card',
            'Personal loan', 
            'Buy Now, Pay Later (BNPL)',
            'Savings account',
            'Money transfers'
        ]
        
        self.product_mapping = {
            'Credit card or prepaid card': 'Credit card',
            'Payday loan, title loan, or personal loan': 'Personal loan',
            'Payday loan, title loan, personal loan, or advance loan': 'Personal loan',
            'Money transfer, virtual currency, or money service': 'Money transfers',
            'Money transfers': 'Money transfers',
            'Checking or savings account': 'Savings account',
            'Bank account or service': 'Savings account',
            'Credit card': 'Credit card',
            'Personal loan': 'Personal loan',
            'Buy Now, Pay Later (BNPL)': 'Buy Now, Pay Later (BNPL)',
            'Savings account': 'Savings account'
        }
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load complaint data from CSV file.
        
        Args:
            file_path: Path to the complaints CSV file
            
        Returns:
            DataFrame with loaded data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Loaded data: {df.shape}")
        return df
    
    def filter_by_products(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data to include only target financial products.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Filtered DataFrame
        """
        # Apply product mapping
        df['Product_mapped'] = df['Product'].map(self.product_mapping)
        df['Product_mapped'] = df['Product_mapped'].fillna('Other')
        
        # Filter for target products
        filtered_df = df[df['Product_mapped'].isin(self.target_products)].copy()
        
        print(f"Filtered by products: {len(df)} -> {len(filtered_df)} complaints")
        return filtered_df
    
    def clean_narrative_text(self, text: str) -> Optional[str]:
        """
        Clean complaint narrative text for better embedding quality.
        
        Args:
            text: Raw complaint narrative
            
        Returns:
            Cleaned text or None if text is too short
        """
        if pd.isna(text):
            return None
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove common boilerplate text
        boilerplate_phrases = [
            "i am writing to file a complaint",
            "i would like to file a complaint",
            "this is a complaint about",
            "dear sir or madam",
            "to whom it may concern",
            "i am contacting you regarding",
            "i am writing this letter to",
            "xxxx", "xx/xx/xxxx"
        ]
        
        for phrase in boilerplate_phrases:
            text = text.replace(phrase, "")
        
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove very short texts (less than 20 characters)
        if len(text) < 20:
            return None
        
        return text
    
    def process_complaints(self, df: pd.DataFrame, narrative_column: str = 'Consumer complaint narrative') -> pd.DataFrame:
        """
        Complete processing pipeline for complaint data.
        
        Args:
            df: Input DataFrame
            narrative_column: Name of the narrative column
            
        Returns:
            Processed DataFrame ready for embedding
        """
        # Filter by products
        df_filtered = self.filter_by_products(df)
        
        # Remove rows without narratives
        df_with_narratives = df_filtered[df_filtered[narrative_column].notna()].copy()
        print(f"With narratives: {len(df_with_narratives)} complaints")
        
        # Clean narratives
        df_with_narratives['cleaned_narrative'] = df_with_narratives[narrative_column].apply(
            self.clean_narrative_text
        )
        
        # Remove rows where cleaning resulted in None
        df_cleaned = df_with_narratives.dropna(subset=['cleaned_narrative']).copy()
        print(f"After cleaning: {len(df_cleaned)} complaints")
        
        # Add text length metrics
        df_cleaned['original_length'] = df_cleaned[narrative_column].str.len()
        df_cleaned['cleaned_length'] = df_cleaned['cleaned_narrative'].str.len()
        
        # Select and rename relevant columns
        columns_to_keep = [
            'Complaint ID',
            'Product_mapped',
            'Issue',
            'Sub-issue',
            'Company',
            'State',
            'Date received',
            'cleaned_narrative',
            'original_length',
            'cleaned_length'
        ]
        
        final_df = df_cleaned[columns_to_keep].copy()
        final_df = final_df.rename(columns={
            'Product_mapped': 'Product',
            'cleaned_narrative': 'Consumer_complaint_narrative'
        })
        
        return final_df
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save processed data to CSV file.
        
        Args:
            df: Processed DataFrame
            output_path: Path to save the file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to: {output_path}")
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics of the processed data.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_complaints': len(df),
            'products': df['Product'].value_counts().to_dict(),
            'avg_narrative_length': df['cleaned_length'].mean(),
            'date_range': {
                'start': df['Date received'].min(),
                'end': df['Date received'].max()
            },
            'top_companies': df['Company'].value_counts().head(10).to_dict(),
            'top_issues': df['Issue'].value_counts().head(10).to_dict()
        }
        
        return summary


def download_nltk_data():
    """Download required NLTK data."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')


if __name__ == "__main__":
    # Example usage
    processor = ComplaintDataProcessor()
    
    # This would be used in a script context
    # df = processor.load_data("../data/complaints.csv")
    # processed_df = processor.process_complaints(df)
    # processor.save_processed_data(processed_df, "../data/filtered_complaints.csv")
    # summary = processor.get_data_summary(processed_df)
    # print(summary)
