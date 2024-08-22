# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : server.py
@Time     : 2024/8/22 10:08
@Author   : Hjw
@License  : (C)Copyright 2018-2025
"""
import io
import json
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # For some reason, transformers decided to use .isin for a simple op, which is not supported on MPS
os.environ["IN_STREAMLIT"] = "true"  # Avoid multiprocessing inside surya
os.environ["PDFTEXT_CPU_WORKERS"] = "1"  # Avoid multiprocessing inside pdftext

import pypdfium2  # Needs to be at the top to avoid warnings
import asyncio
import argparse
import time
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import concurrent.futures
from marker.convert import convert_single_pdf  # Import function to parse PDF
# from marker.logger import configure_logging  # Import logging configuration
from marker.models import load_all_models  # Import function to load models
from marker.output import save_markdown, get_markdown_filepath
from marker.settings import settings  # Import settings
from contextlib import asynccontextmanager
# import logging
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile

# Initialize logging
# configure_logging()
# logger = logging.getLogger(__name__)

# Global variable to hold model list
MODEL_LIST = None
OUTPUT_FOLDER = 'output'


# Event that runs on startup to load all models
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        mp.set_start_method('spawn')  # Required for CUDA, forkserver doesn't work
    except RuntimeError:
        raise RuntimeError("Set start method to spawn twice. This may be a temporary issue with the script. Please try running it again.")

    global MODEL_LIST
    if settings.TORCH_DEVICE == "mps" or settings.TORCH_DEVICE_MODEL == "mps":
        print(
            "Cannot use MPS with torch multiprocessing share_memory. This will make things less memory efficient. If you want to share memory, you have to use CUDA or CPU.  Set the TORCH_DEVICE environment variable to change the device.")

        MODEL_LIST = None
    else:
        MODEL_LIST = load_all_models()
        for model in MODEL_LIST:
            if model is None:
                continue
            model.share_memory()
    yield


def worker_init(shared_model):
    if shared_model is None:
        shared_model = load_all_models()

    global model_refs
    model_refs = shared_model


def worker_exit():
    global model_refs
    del model_refs


# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


def convert_single_file(filepath: str, output_folder: str = OUTPUT_FOLDER, **kwargs):
    if not filepath:
        raise HTTPException(status_code=400, detail="No file provided")
    filename = os.path.basename(filepath)
    markdown_filepath = get_markdown_filepath(output_folder, filename)
    if os.path.exists(markdown_filepath):
        markdown_text = open(markdown_filepath, "r", encoding='utf-8').read()
        out_meta_filepath = markdown_filepath.rsplit(".", 1)[0] + "_meta.json"
        if os.path.exists(out_meta_filepath):
            metadata = json.load(open(out_meta_filepath, "r"))
        else:
            metadata = {}

        return {
            "filename": filename,
            "markdown": markdown_text,
            "markdown_filepath": markdown_filepath,
            "metadata": metadata,
            "metadata_filepath": out_meta_filepath,
            "status": "ok",
            "time": 0
        }

    entry_time = time.time()
    print(f"Entry time for {filename}: {entry_time}")
    try:
        markdown_text, image_data, metadata = convert_single_pdf(filepath, MODEL_LIST, **kwargs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {filename}. \n{e}")
    completion_time = time.time()
    print(f"Model processes complete time for {filename}: {completion_time}")
    time_difference = completion_time - entry_time
    print(f"Time taken to process {filename}: {time_difference}")

    if len(markdown_text.strip()) < 0:
        raise HTTPException(status_code=400, detail=f"Empty file: {filename}")

    subfolder_path = save_markdown(output_folder, filename, markdown_text, image_data, metadata)
    markdown_filepath = get_markdown_filepath(output_folder, filename)
    out_meta_filepath = markdown_filepath.rsplit(".", 1)[0] + "_meta.json"

    return {
        "filename": filename,
        "markdown": markdown_text,
        "markdown_filepath": markdown_filepath,
        "metadata": metadata,
        "metadata_filepath": out_meta_filepath,
        "status": "ok",
        "time": 0
    }


# Root endpoint to check server status
@app.get("/")
def server():
    """
    Root endpoint to check server status.

    Returns:
    dict: A welcome message.
    """
    return {"message": "Welcome to Marker-api"}


# Endpoint to convert a single PDF to markdown
@app.post("/convert")
def convert_file_to_markdown(
        file: UploadFile = None,
        filepath: str = None,
        output_folder: str = None,
        max_pages: int = None,
        start_page: int = None,
        metadata: Optional[dict] = None,
        langs: Optional[list[str]] = None,
        batch_multiplier: int = 1,
        ocr_all_pages: bool = False):
    """
    Endpoint to convert a single PDF to markdown.

    Args:
    file (UploadFile): The uploaded PDF file.
    filepath (str): The path to the PDF file.
    output_folder (str): The folder to save the markdown file and images.
    **kwargs: Additional keyword arguments.

    Returns:
    dict: The response from processing the PDF file.
    """
    kwargs = {
        "max_pages": max_pages,
        "start_page": start_page,
        "metadata": metadata,
        "langs": langs,
        "batch_multiplier": batch_multiplier,
        "ocr_all_pages": ocr_all_pages
    }
    if file:
        with tempfile.NamedTemporaryFile('w+b', suffix=".pdf") as temp_pdf:
            temp_pdf.write(file.read())
            temp_pdf.seek(0)
            filepath = temp_pdf.name
            convert_single_file(filepath, output_folder, **kwargs)

    return convert_single_file(filepath, output_folder, **kwargs)


# Endpoint to convert multiple PDFs to markdown
@app.post("/batch_convert")
async def convert_files_to_markdown(
        files: List[UploadFile] = None,
        filepaths: list[str] = None,
        output_folder: str = None,
        workers: int = 4,
):
    """
    Endpoint to convert multiple PDFs to markdown.

    Args:
    pdf_files (List[UploadFile]): The list of uploaded PDF files.

    Returns:
    list: The responses from processing each PDF file.
    """

    if not filepaths and not files:
        raise HTTPException(status_code=400, detail="No files provided")
    if not files:
        filenames = [os.path.basename(filepath) for filepath in filepaths]
        print(f"Received files: {filenames}")
        files = [None for _ in range(len(filepaths))]
    else:
        filenames = [file.filename for file in files]
        print(f"Received files: {filenames}")
        filepaths = ['' for _ in range(len(files))]

    assert len(files) == len(filepaths), "Number of files and filepaths do not match"
    print(f"Received {len(files)} files for batch conversion")
    time.sleep(10)
    print("filepaths: ", filepaths)

    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_file = {executor.submit(convert_file_to_markdown, file, filepath, output_folder): file for file, filepath in zip(files, filepaths)}
        for future in as_completed(future_to_file):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append({
                    "filename": future_to_file[future].filename,
                    "status": "error",
                    "error": str(e)
                })

    return results


@app.post("/batch_convert2")
async def batch_convert_files_to_markdown2(
        files: List[UploadFile],
        filepaths: list[str] = None,
        output_folder: str = None,
        max_pages: int = None,
        start_page: int = None,
        metadata: Optional[dict] = None,
        langs: Optional[list[str]] = None,
        batch_multiplier: int = 1,
        ocr_all_pages: bool = False,
        max_workers: int = 4):
    """
    Endpoint to convert multiple PDFs to markdown using multithreading.

    Args:
    files (List[UploadFile]): List of uploaded PDF files.
    output_folder (str): The folder to save markdown files.
    max_pages (int): Maximum number of pages to process.
    start_page (int): Page number to start processing.
    metadata (dict): Optional metadata for processing.
    langs (list[str]): Optional list of languages.
    batch_multiplier (int): Batch processing multiplier.
    ocr_all_pages (bool): Whether to OCR all pages or not.
    max_workers (int): Maximum number of threads to use for parallel processing.

    Returns:
    dict: A response containing the status and details of each processed file.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    async def process_file(file: UploadFile, model_list, max_pages, start_page, metadata, langs, batch_multiplier, ocr_all_pages):
        try:
            filename = file.filename
            print(f"Processing file: {filename}")
            file_content = await file.read()

            entry_time = time.time()
            markdown_text, image_data, metadata = convert_single_pdf(file_content, model_list,
                                                                     max_pages, start_page, metadata, langs, batch_multiplier, ocr_all_pages)
            completion_time = time.time()
            time_difference = completion_time - entry_time
            print(f"Time taken to process {filename}: {time_difference}")

            if output_folder:
                if len(markdown_text.strip()) <= 0:
                    raise HTTPException(status_code=400, detail=f"Empty file: {filename}")

                subfolder_path = save_markdown(output_folder, filename, markdown_text, image_data, metadata)
                markdown_filepath = get_markdown_filepath(output_folder, filename)
                return {
                    "filename": filename,
                    "markdown": markdown_filepath,
                    "output_folder": subfolder_path,
                    "status": "ok",
                    "time": time_difference
                }

            for i, (img_filename, image) in enumerate(image_data.items()):
                image_io = io.BytesIO()
                image.save(image_io, format='PNG')
                image_bytes = image_io.getvalue()
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                image_data[f'{img_filename}'] = image_base64

            return {
                "filename": filename,
                "markdown": markdown_text,
                "metadata": metadata,
                "images": image_data,
                "status": "ok",
                "time": time_difference
            }

        except Exception as e:
            return {
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            }

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_file, file, MODEL_LIST, max_pages, start_page, metadata, langs, batch_multiplier, ocr_all_pages): file for
                          file in files}
        for future in as_completed(future_to_file):
            result = future.result()
            results.append(result)

    return {"results": results}


# Main function to run the server
def main():
    parser = argparse.ArgumentParser(description="Run the marker-api server.")
    parser.add_argument("--host", default="0.0.0.0", help="Host IP address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    # parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    # if args.debug:
    #     logger.setLevel(logging.DEBUG)

    import uvicorn
    uvicorn.run("server:app", host=args.host, port=args.port)


# Entry point to start the server
if __name__ == "__main__":
    main()
