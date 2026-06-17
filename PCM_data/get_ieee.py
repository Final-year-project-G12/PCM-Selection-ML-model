import urllib.request
import re
import json

url = 'https://ieeexplore.ieee.org/document/11141790'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})

try:
    with urllib.request.urlopen(req) as response:
        html = response.read().decode('utf-8', errors='ignore')

        title_match = re.search(r'"title":"(.*?)","', html)
        if title_match:
            print("Title:", title_match.group(1))
            
        abstract_match = re.search(r'"abstract":"(.*?)","', html)
        if abstract_match:
            print("Abstract:", abstract_match.group(1))
        else:
            print("Could not find xplGlobal.document.metadata in HTML.")
            
        print("\n--- Raw Text Snippets ---")
        import html.parser
        class HTMLFilter(html.parser.HTMLParser):
            text = []
            def handle_data(self, data):
                self.text.append(data.strip())
        f = HTMLFilter()
        f.feed(html)
        full_text = " ".join([t for t in f.text if t])
        
        import re
        sentences = re.split(r'(?<=[.!?]) +', full_text)
        for s in sentences:
            if re.search(r'imput|missing|preprocess|process|KNN|normalize|interp', s, re.IGNORECASE):
                print("-", s)

except Exception as e:
    print("Request failed:", e)
