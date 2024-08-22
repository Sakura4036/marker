# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : server.py
@Time     : 2024/8/22 10:08
@Author   : Hjw
@License  : (C)Copyright 2018-2025
"""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # For some reason, transformers decided to use .isin for a simple op, which is not supported on MPS
os.environ["IN_STREAMLIT"] = "true" # Avoid multiprocessing inside surya
os.environ["PDFTEXT_CPU_WORKERS"] = "1" # Avoid multiprocessing inside pdftext

import pypdfium2 # Needs to be at the top to avoid warnings
import asyncio
import argparse
import time
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import concurrent.futures
from marker.convert import convert_single_pdf  # Import function to parse PDF
from marker.logger import configure_logging  # Import logging configuration
from marker.models import load_all_models  # Import function to load models
from marker.output import save_markdown, get_markdown_filepath, markdown_exists
from marker.settings import settings  # Import settings
from convert import process_single_pdf  # Import function to parse PDF
from contextlib import asynccontextmanager
import logging
import torch.multiprocessing as mp
from tqdm import tqdm

# Initialize logging
configure_logging()
logger = logging.getLogger(__name__)

# Global variable to hold model list
model_list = None



# Event that runs on startup to load all models
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        mp.set_start_method('spawn')  # Required for CUDA, forkserver doesn't work
    except RuntimeError:
        raise RuntimeError("Set start method to spawn twice. This may be a temporary issue with the script. Please try running it again.")

    global model_list
    if settings.TORCH_DEVICE == "mps" or settings.TORCH_DEVICE_MODEL == "mps":
        print(
            "Cannot use MPS with torch multiprocessing share_memory. This will make things less memory efficient. If you want to share memory, you have to use CUDA or CPU.  Set the TORCH_DEVICE environment variable to change the device.")

        model_list = None
    else:
        model_list = load_all_models()
        for model in model_list:
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


def convert_single_file(filepath: str = None,
                        save_path: str = None,
                        file: UploadFile = None,
                        max_pages: int = None,
                        start_page: int = None,
                        metadata: Optional[dict] = None,
                        langs: Optional[list[str]] = None,
                        batch_multiplier: int = 1,
                        ocr_all_pages: bool = False):
    pass


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
async def convert_file_to_markdown(filepath: str = None,
                                   output_folder: str = None,
                                   file: UploadFile = None,
                                   max_pages: int = None,
                                   start_page: int = None,
                                   metadata: Optional[dict] = None,
                                   langs: Optional[list[str]] = None,
                                   batch_multiplier: int = 1,
                                   ocr_all_pages: bool = False):
    """
    Endpoint to convert a single PDF to markdown.

    Args:
    filepath (str): The path to the PDF file.
    save_type (str): "text" means only save the Markdown text to a file, "all" means save as all markdown, None or others string means do not save.
    save_path (str): The path to save the file.
    file (UploadFile): The uploaded PDF file.
    **kwargs: Additional keyword arguments.

    Returns:
    dict: The response from processing the PDF file.
    """
    if not filepath and not file:
        raise HTTPException(status_code=400, detail="No file provided")
    if not file:
        filename = os.path.basename(filepath)
        logger.debug(f"Received file: {filepath}")
        file_content = filepath
    else:
        logger.debug(f"Received file: {file.filename}")
        filename = file.filename
        file_content = await file.read()
    if output_folder:
        markdown_filepath = get_markdown_filepath(output_folder, filename)
        if markdown_exists(markdown_filepath):
            subfolder_path = os.path.dirname(markdown_filepath)
            return {
                "filename": filename,
                "markdown": markdown_filepath,
                "output_folder": subfolder_path,
                "status": "ok",
                "time": 0
            }

    entry_time = time.time()
    logger.debug(f"Entry time for {filename}: {entry_time}")
    try:
        markdown_text, image_data, metadata = convert_single_pdf(file_content, model_list,
                                                                 max_pages, start_page, metadata, langs, batch_multiplier, ocr_all_pages)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {filename}. \n{e}")
    completion_time = time.time()
    logger.debug(f"Model processes complete time for {filename}: {completion_time}")
    time_difference = completion_time - entry_time
    logger.debug(f"Time taken to process {filename}: {time_difference}")
    if output_folder:
        if len(markdown_text.strip()) < 0:
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

    return {
        "filename": filename,
        "markdown": markdown_text,
        "metadata": metadata,
        "images": image_data,
        "status": "ok",
        "time": time_difference
    }


# Endpoint to convert multiple PDFs to markdown
@app.post("/batch_convert")
async def convert_files_to_markdown(filepaths: list[str] = None,
                                    files: List[UploadFile] = None,
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
    logger.info(f"Received {len(files)} files for batch conversion")

    if not filepaths and not files:
        raise HTTPException(status_code=400, detail="No files provided")
    if not files:
        filenames = [os.path.basename(filepath) for filepath in filepaths]
        logger.debug(f"Received files: {filenames}")
        files = [None for _ in range(len(filepaths))]
    else:
        filenames = [file.filename for file in files]
        logger.debug(f"Received files: {filenames}")
        filepaths = [None for _ in range(len(files))]
    # task_args = [(f, output_folder, None, min_length) for f in file_contents]
    # total_processes = min(len(task_args), workers)
    # with mp.Pool(processes=total_processes, initializer=worker_init, initargs=(model_list,)) as pool:
    #     list(tqdm(pool.imap(process_single_pdf, task_args), total=len(task_args), desc="Processing PDFs", unit="pdf"))
    #
    #     pool._worker_handler.terminate = worker_exit

    async def process_files(filepaths, files):
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            coroutines = [
                loop.run_in_executor(pool, convert_file_to_markdown, filepaths, output_folder, file)
                for filepath, file in zip(filepaths, files)
            ]
            return await asyncio.gather(*coroutines)

    entry_time = time.time()
    logger.debug(f"Entry time : {entry_time}")

    responses = await process_files(filepaths, files)

    completion_time = time.time()
    logger.debug(f"Model processes complete time : {completion_time}")
    time_difference = completion_time - entry_time
    logger.debug(f"Time taken: {time_difference}")

    return responses


# Main function to run the server
def main():
    parser = argparse.ArgumentParser(description="Run the marker-api server.")
    parser.add_argument("--host", default="0.0.0.0", help="Host IP address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    import uvicorn
    uvicorn.run("server:app", host=args.host, port=args.port)


# Entry point to start the server
if __name__ == "__main__":
    main()
