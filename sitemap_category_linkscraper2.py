import requests
from bs4 import BeautifulSoup

# Target cateogry URL (tag cateogry)
sitemap_url = "https://www.wired.com/categories-sitemap.xml?hierarchy=channels"

# Send request
response = requests.get(sitemap_url)
response.raise_for_status()  # Check request status

# Parse XML
soup = BeautifulSoup(response.content, 'xml')

# Extract text content of all <loc> tags
tag_urls = [loc.text.strip() for loc in soup.find_all('loc')]

# Print first few links and total count
for url in tag_urls[:10]:
    print(url)

print(f"\nTotal tag pages found: {len(tag_urls)}")

# Save as txt file
with open("../cateogry/wired_category_csv/wired_tag_urls.txt", "w") as f:
    for url in tag_urls:
        f.write(url + "\n")
