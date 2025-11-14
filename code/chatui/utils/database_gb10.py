# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GB10-Optimized Database Module
===============================

This version uses LOCAL embeddings on GB10 GPU instead of cloud embeddings.

Key Changes:
- Uses gb10_embeddings.get_embeddings() for smart embedding selection
- Defaults to local embeddings (can be overridden with USE_LOCAL_EMBEDDINGS=false)
- Maintains full compatibility with original database.py API

Author: NVIDIA
Date: 2025-11-14
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    WebBaseLoader,
    UnstructuredPDFLoader,
    TextLoader,
    CSVLoader
)
from langchain_community.vectorstores import Chroma

# MODIFIED: Use GB10 smart embeddings instead of always cloud
from chatui.utils.gb10_embeddings import get_embeddings

from typing import Any, Dict, List, Tuple, Union
from urllib.parse import urlparse
import os
import shutil
import mimetypes


# GB10 Configuration
# Set USE_LOCAL_EMBEDDINGS=true to use GB10 GPU embeddings (default for GB10)
# Set USE_LOCAL_EMBEDDINGS=false to use cloud embeddings
USE_LOCAL_EMBEDDINGS = os.getenv('USE_LOCAL_EMBEDDINGS', 'true').lower() in ['true', '1', 'yes']

if USE_LOCAL_EMBEDDINGS:
    print("[config] Using LOCAL embeddings (GB10 GPU)")
    EMBEDDINGS_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    print(f"[config] Embedding model: {EMBEDDINGS_MODEL}")
    print(f"[config] Device: {os.getenv('EMBEDDING_DEVICE', 'cuda')}")
else:
    print("[config] Using CLOUD embeddings (NVIDIA API)")
    INTERNAL_API = os.getenv('INTERNAL_API', 'no')
    if INTERNAL_API == 'yes':
        EMBEDDINGS_MODEL = 'nvdev/nvidia/nv-embedqa-e5-v5'
        print(f"[config] Using internal embedding model: {EMBEDDINGS_MODEL}")
    else:
        EMBEDDINGS_MODEL = 'nvidia/nv-embedqa-e5-v5'
        print(f"[config] Using public embedding model: {EMBEDDINGS_MODEL}")


# Set the chunk size and overlap for the text splitter
DEFAULT_CHUNK_SIZE = 250
DEFAULT_CHUNK_OVERLAP = 0

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", DEFAULT_CHUNK_SIZE))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP))


# Adding nltk data
import nltk

def download_nltk_if_missing():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')

download_nltk_if_missing()


# Functions for dealing with URLs
def is_valid_url(url: str) -> bool:
    """ This is a helper function for checking if the URL is valid. It isn't fail proof, but it will catch most common errors. """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def safe_load(url):
    """ This is a helper function for loading the URL. It protects against false negatives from is_value_url and
        filters for actual web pages. Returns None if it fails.
    """
    try:
        return WebBaseLoader(url).load()
    except Exception as e:
        print(f"[upload] Skipping {url}: {e}")
        return None


def upload(urls: List[str]):
    """ This is a helper function for parsing the user inputted URLs and uploading them into the vector store. """

    urls = [url for url in urls if is_valid_url(url)]

    docs = []
    for url in urls:
        result = safe_load(url)
        if result is not None:
            docs.append(result)

    docs_list = [item for sublist in docs for item in sublist]

    if not docs_list:
        # If no documents were loaded, return None
        print("[upload] No URLs provided.")
        return None

    try:
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=CHUNK_SIZE, chunk_overlap=0
        )
        doc_splits = text_splitter.split_documents(docs_list)

        # MODIFIED: Use GB10 smart embeddings
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=get_embeddings(),  # Auto-selects local or cloud based on config
            persist_directory="/project/data",
        )
        return vectorstore

    except Exception as e:
        print(f"[upload] Vectorstore creation failed: {e}")
        return None


# Functions for dealing with file uploads/embeddings

## Helper functions

def load_documents_from_files(file_paths: List[str]) -> List[Any]:
    """Load and return documents from supported file types."""
    docs = []

    for fpath in file_paths:
        ext = os.path.splitext(fpath)[-1].lower()

        loader_cls = {
            ".pdf": UnstructuredPDFLoader,
            ".txt": TextLoader,
            ".md": TextLoader,
            ".csv": CSVLoader
        }.get(ext)

        if loader_cls is None:
            print(f"[load_documents] Skipping unsupported file type: {fpath}")
            continue

        try:
            loaded = loader_cls(fpath).load()
            docs.append(loaded)
        except Exception as e:
            print(f"[load_documents] Failed to load {fpath}: {e}")

    return [item for sublist in docs for item in sublist]


def split_documents(docs: List[Any]):
    """Split documents into smaller chunks using recursive splitter."""
    print(f"[split_documents] Splitting {len(docs)} docs with chunk size {CHUNK_SIZE}, overlap {CHUNK_OVERLAP}")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)

def embed_documents(doc_splits: List[Any]):
    """Embed and store the split documents into Chroma vectorstore."""
    try:
        print(f"[embed_documents] Embedding {len(doc_splits)} chunks")
        if USE_LOCAL_EMBEDDINGS:
            print(f"[embed_documents] Using LOCAL embeddings on GB10 GPU")
        else:
            print(f"[embed_documents] Using CLOUD embeddings: {EMBEDDINGS_MODEL}")

        # MODIFIED: Use GB10 smart embeddings
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=get_embeddings(),  # Auto-selects local or cloud based on config
            persist_directory="/project/data",
        )
        return vectorstore

    except Exception as e:
        print(f"[embed_documents] Vectorstore creation failed: {e}")
        return None


## Main function that use helper functions

def upload_files(file_paths: List[str]):
    """Upload files into the vector store pipeline."""

    if not file_paths:
        print("[upload_files] No documents provided.")
        return None

    try:
        docs_list = load_documents_from_files(file_paths)

        if not docs_list:
            print("[upload_files] No documents successfully loaded.")
            return None

        doc_splits = split_documents(docs_list)
        return embed_documents(doc_splits)

    except Exception as e:
        print(f"[upload_files] Pipeline failed: {e}")
        return None




def _clear(
    persist_directory: str = "/project/data",
    collection_name: str = "rag-chroma",
    delete_all: bool = True
):
    """Clear the Chroma collection and optionally delete all shard folders (excluding hidden files)."""
    try:
        # MODIFIED: Use GB10 smart embeddings
        # Clear the collection via Chroma client
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=get_embeddings(),  # Auto-selects local or cloud based on config
            persist_directory=persist_directory,
        )
        vectorstore._client.delete_collection(name=collection_name)
        vectorstore._client.create_collection(name=collection_name)
        print(f"[clear] Collection '{collection_name}' cleared.")

        if delete_all:
            for item in os.listdir(persist_directory):
                if item.startswith(".") or item.startswith("chroma."):
                    continue  # Skip hidden files like .gitkeep and the sqlite3 file

                path = os.path.join(persist_directory, item)
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                        print(f"[clear] Removed file: {item}")
                    elif os.path.isdir(path):
                        shutil.rmtree(path)
                        print(f"[clear] Removed directory: {item}")
                except Exception as file_err:
                    print(f"[clear] Could not delete {item}: {file_err}")

    except Exception as e:
        print(f"[clear] Failed to clear vector store: {e}")


def get_retriever():
    """ This is a helper function for returning the retriever object of the vector store. """
    # MODIFIED: Use GB10 smart embeddings
    vectorstore = Chroma(
        collection_name="rag-chroma",
        embedding_function=get_embeddings(),  # Auto-selects local or cloud based on config
        persist_directory="/project/data",
    )
    retriever = vectorstore.as_retriever()
    return retriever
