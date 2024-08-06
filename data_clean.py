import requests
from bs4 import BeautifulSoup
import re

def collect_and_clean_data(url):
    # Fetch the webpage
    response = requests.get(url)
    
    # Parse HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract text and remove HTML tags
    text = soup.get_text()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove very short lines (e.g., less than 20 characters)
    lines = [line for line in text.split('\n') if len(line) > 20]
    
    return '\n'.join(lines)

# Example usage
url = "https://en.wikipedia.org/wiki/Machine_learning"
cleaned_text = collect_and_clean_data(url)
print(cleaned_text[:500])  # Print first 500 characters as a sample
