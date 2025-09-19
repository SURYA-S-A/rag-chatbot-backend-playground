import os
from typing import List, Optional
from fastapi import UploadFile
import fitz
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import gc


class FileProcessor:
    def __init__(
        self, chunk_size=1000, chunk_overlap=200, max_files=5, max_mb=10
    ) -> None:
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
        )
        self.max_files = max_files
        self.max_bytes = max_mb * 1024 * 1024  # MB → bytes

    def get_file_name(self, file_path: str) -> str:
        """Generate safe filename (without spaces, lowercase)."""
        base = os.path.basename(file_path)
        name = os.path.splitext(base)[0]
        return name.lower().replace(" ", "_")

    def validate_files(
        self,
        files: Optional[List[UploadFile]] = None,
        file_paths: Optional[List[str]] = None,
    ) -> None:
        """Ensure number and size constraints for both upload files and file paths."""

        if files:
            if len(files) > self.max_files:
                raise ValueError(f"Too many files! Limit = {self.max_files}")

            # For UploadFile objects, check size if available
            for file in files:
                if hasattr(file, "size") and file.size and file.size > self.max_bytes:
                    raise ValueError(
                        f"File {file.filename} exceeds {self.max_bytes/1024/1024:.1f} MB"
                    )

        elif file_paths:
            if len(file_paths) > self.max_files:
                raise ValueError(f"Too many files! Limit = {self.max_files}")

            for fp in file_paths:
                if os.path.getsize(fp) > self.max_bytes:
                    raise ValueError(
                        f"File {fp} exceeds {self.max_bytes/1024/1024:.1f} MB"
                    )

    async def _load_pdf_from_memory(
        self, filename: str, content: bytes
    ) -> List[Document]:
        """Load PDF from memory bytes with proper cleanup."""
        doc = None
        try:
            doc = fitz.open("pdf", content)
            documents = []

            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()

                document = Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "page": page_num,
                        "filename": self.get_file_name(filename),
                    },
                )
                documents.append(document)

            return documents

        except Exception as e:
            raise ValueError(f"Error processing PDF {filename}: {str(e)}")
        finally:
            # Close document to free PyMuPDF internal buffers
            if doc is not None:
                doc.close()

    async def load_and_split(
        self,
        files: Optional[List[UploadFile]] = None,
        file_paths: Optional[List[str]] = None,
    ) -> List[Document]:
        """Load and split PDFs with comprehensive memory cleanup."""

        self.validate_files(files=files, file_paths=file_paths)
        all_docs = []

        try:
            if files:
                for file in files:
                    content = None
                    docs = None
                    split_docs = None

                    try:
                        # Read file content
                        content = await file.read()

                        # Validate size
                        if len(content) > self.max_bytes:
                            raise ValueError(
                                f"File {file.filename} exceeds {self.max_bytes/1024/1024:.1f} MB"
                            )

                        # Process PDF (document cleanup handled in _load_pdf_from_memory)
                        docs = await self._load_pdf_from_memory(file.filename, content)

                        # Split documents
                        split_docs = self.text_splitter.split_documents(docs)

                        # Add metadata
                        file_name = self.get_file_name(file.filename)
                        for d in split_docs:
                            d.metadata.update({"filename": file_name})

                        print(f"{file_name}: {len(split_docs)} chunks")
                        all_docs.extend(split_docs)

                        # Reset file pointer
                        await file.seek(0)

                    finally:
                        # Clean up intermediate objects
                        del content, docs, split_docs
                        # Force garbage collection
                        gc.collect()

            elif file_paths:
                for file_path in file_paths:
                    loader = None
                    docs = None
                    split_docs = None

                    try:
                        loader = PyMuPDFLoader(file_path)
                        docs = loader.load()
                        split_docs = self.text_splitter.split_documents(docs)

                        file_name = self.get_file_name(file_path)
                        for d in split_docs:
                            d.metadata.update({"filename": file_name})

                        print(f"{file_name}: {len(split_docs)} chunks")
                        all_docs.extend(split_docs)

                    finally:
                        # Clean up file-based processing
                        del loader, docs, split_docs
                        gc.collect()

            return all_docs

        except Exception as e:
            # Clean up on error
            gc.collect()
            raise e
