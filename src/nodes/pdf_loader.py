import os
import re
import json
import time
import hashlib
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


class PDFCrawlerLoader:
    """
    Crawl websites for PDFs, download them,
    extract text, and store structured data.
    """

    def __init__(
        self,
        base_urls,
        download_dir="downloads",
        output_file="documents.jsonl",
        timeout=20,
        max_pdfs=100,
    ):
        self.base_urls = base_urls
        self.download_dir = download_dir
        self.output_file = output_file
        self.timeout = timeout
        self.max_pdfs = max_pdfs

        os.makedirs(download_dir, exist_ok=True)

        self.visited = set()
        self.pdf_links = set()

    # -----------------------------
    # Crawl webpages for PDFs
    # -----------------------------
    def crawl(self):
        for url in self.base_urls:
            print(f"\n[CRAWLING] {url}")
            self._crawl_page(url)

        print(f"\n[FOUND PDFs] {len(self.pdf_links)}")

    def _crawl_page(self, url):
        if url in self.visited:
            return

        self.visited.add(url)

        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            for link in soup.find_all("a", href=True):
                href = link["href"]
                full_url = urljoin(url, href)

                # PDF link
                if full_url.lower().endswith(".pdf"):
                    self.pdf_links.add(full_url)

                    if len(self.pdf_links) >= self.max_pdfs:
                        return

                # Internal links only
                elif self._is_same_domain(url, full_url):
                    if full_url not in self.visited:
                        try:
                            self._crawl_page(full_url)
                        except Exception:
                            pass

        except Exception as e:
            print(f"[ERROR] {url} -> {e}")

    # -----------------------------
    # Download PDFs
    # -----------------------------
    def download_pdfs(self):
        downloaded_files = []

        for pdf_url in self.pdf_links:
            try:
                filename = self._generate_filename(pdf_url)
                filepath = os.path.join(self.download_dir, filename)

                if os.path.exists(filepath):
                    print(f"[SKIP] Already exists: {filename}")
                    downloaded_files.append(filepath)
                    continue

                print(f"[DOWNLOADING] {pdf_url}")

                response = requests.get(pdf_url, timeout=self.timeout)
                response.raise_for_status()

                with open(filepath, "wb") as f:
                    f.write(response.content)

                downloaded_files.append(filepath)
                time.sleep(1)

            except Exception as e:
                print(f"[DOWNLOAD ERROR] {pdf_url} -> {e}")

        return downloaded_files

    # -----------------------------
    # Extract PDF text
    # -----------------------------
    def extract_text(self, pdf_path):
        try:
            reader = PdfReader(pdf_path)
            text = ""

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            return self.clean_text(text)

        except Exception as e:
            print(f"[PDF ERROR] {pdf_path} -> {e}")
            return ""

    # -----------------------------
    # Store structured data
    # -----------------------------
    def build_dataset(self, pdf_files):
        with open(self.output_file, "a", encoding="utf-8") as outfile:
            for pdf_file in pdf_files:
                text = self.extract_text(pdf_file)

                if not text.strip():
                    continue

                chunks = self.semantic_chunk(text)

                for idx, chunk in enumerate(chunks):
                    record = {
                        "id": self.generate_id(pdf_file, idx),
                        "source": pdf_file,
                        "chunk_id": idx,
                        "text": chunk,
                        "metadata": {
                            "domain": self.detect_domain(pdf_file),
                            "type": "pdf_document",
                        },
                    }

                    outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"\n[DATASET SAVED] {self.output_file}")

    # -----------------------------
    # Semantic chunking
    # -----------------------------
    def semantic_chunk(self, text, chunk_size=1200):
        paragraphs = text.split("\n\n")

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()

            if not para:
                continue

            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += "\n" + para
            else:
                chunks.append(current_chunk.strip())
                current_chunk = para

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    # -----------------------------
    # Utility functions
    # -----------------------------
    def clean_text(self, text):
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def detect_domain(self, path):
        filename = os.path.basename(path).lower()

        if "physics" in filename:
            return "physics"
        elif "math" in filename:
            return "mathematics"
        elif "biology" in filename:
            return "biology"
        elif "chemistry" in filename:
            return "chemistry"

        return "general"

    def generate_id(self, source, idx):
        raw = f"{source}_{idx}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _generate_filename(self, url):
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)

        if not filename.endswith(".pdf"):
            filename += ".pdf"

        return filename

    def _is_same_domain(self, base_url, target_url):
        return urlparse(base_url).netloc == urlparse(target_url).netloc


# =====================================
# Example Usage
# =====================================
if __name__ == "__main__":
    BASE_URLS = [
        "https://ncert.nic.in/textbook.php",
    ]

    loader = PDFCrawlerLoader(
        base_urls=BASE_URLS,
        download_dir="pdf_downloads",
        output_file="education_dataset.jsonl",
        max_pdfs=20,
    )

    # Step 1: Crawl for PDFs
    loader.crawl()

    # Step 2: Download PDFs
    pdf_files = loader.download_pdfs()

    # Step 3: Extract + Chunk + Store
    loader.build_dataset(pdf_files)

    print("\nPipeline completed successfully.")
