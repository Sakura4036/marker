# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
FastAPI server for marker PDF conversion service.
支持单文件和批量文件转换，可配置GPU数量和workers数量。
"""
import json
import os
import tempfile
import time
from typing import List, Optional, Dict
import torch
import torch.multiprocessing as mp
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from marker.config.parser import ConfigParser
from marker.models import create_model_dict
from marker.output import get_markdown_filepath, save_output
from marker.settings import settings

# 全局变量
OUTPUT_FOLDER = 'output'
model_refs = None  # 用于在进程间共享模型


def create_shared_models():
    """
    创建可在进程间共享的模型
    确保在主进程中正确初始化CUDA设备
    """
    try:
        # 获取可用的GPU数量
        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            print("No CUDA devices available. Using CPU.")
            return create_model_dict(device="cpu")
        else:
            print(f"Found {n_gpus} CUDA devices")
            
        # 始终使用 GPU 0 来避免设备间共享内存的问题
        gpu_id = 0
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
        print(f"Initializing models on {device}")
        
        models = create_model_dict(device=device)
        # 确保所有模型都在共享内存中
        for model in models.values():
            if model is not None:
                model.model.share_memory()
        return models
    except Exception as e:
        print(f"Error creating shared models: {e}")
        print("Falling back to CPU")
        return create_model_dict(device="cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    服务启动时加载模型
    支持多GPU配置，确保正确的设备初始化
    """
    try:
        # 设置多进程启动方式
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        print("Warning: spawn start method already set")
        
    global model_refs
    if settings.TORCH_DEVICE == "mps":
        print("Cannot use MPS with torch multiprocessing share_memory. Using CPU instead.")
        model_refs = create_model_dict(device="cpu")
    else:
        model_refs = create_shared_models()
        
    yield
    
    # 清理模型
    if model_refs:
        del model_refs


# 初始化FastAPI
app = FastAPI(lifespan=lifespan)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


def convert_single_file(
        filepath: str,
        output_folder: str = OUTPUT_FOLDER,
        max_pages: int = None,
        start_page: int = None,
        metadata: Optional[dict] = None,
        langs: Optional[List[str]] = None,
        batch_multiplier: int = 1,
        ocr_all_pages: bool = False,
        use_llm: bool = False,
        output_format: str = "markdown",
) -> Dict:
    """
    转换单个PDF文件
    
    Args:
        filepath: PDF文件路径
        output_folder: 输出文件夹
        max_pages: 最大页数
        start_page: 起始页数
        metadata: 元数据
        langs: 语言列表
        batch_multiplier: 批处理倍数
        ocr_all_pages: 是否OCR所有页
        use_llm: 是否使用LLM
        output_format: 输出格式，默认为markdown，支持markdown和json，html
    """
    if not filepath:
        raise HTTPException(status_code=400, detail="No file provided")
    if not output_folder:
        output_folder = OUTPUT_FOLDER
        os.makedirs(output_folder, exist_ok=True)
    # 配置转换参数
    config = {
        "output_dir": output_folder,
        "max_pages": max_pages,
        "start_page": start_page,
        "metadata": metadata,
        "languages": langs,
        "batch_multiplier": batch_multiplier,
        "force_ocr": ocr_all_pages,
        "use_llm": use_llm,
        "output_format": output_format
    }

    print("Converting file: ", filepath)
    print("Output folder: ", output_folder)
    filename = os.path.basename(filepath)
    markdown_filepath = get_markdown_filepath(output_folder, filename)
    out_meta_filepath = markdown_filepath.rsplit(".", 1)[0] + "_meta.json"
    print("Check Markdown file path: ", markdown_filepath)

    # 检查是否已存在转换结果
    if os.path.exists(markdown_filepath):
        print(f"File {filename} already exists. Returning existing file.")
        markdown_text = open(markdown_filepath, "r", encoding='utf-8').read()
        # metadata = json.load(open(out_meta_filepath, "r")) if os.path.exists(out_meta_filepath) else {}

        return {
            "filename": filename,
            "markdown": markdown_text,
            "markdown_filepath": markdown_filepath,
            # "metadata": metadata,
            "metadata_filepath": out_meta_filepath,
            "status": "ok",
            "time": 0
        }

    # 执行转换
    entry_time = time.time()
    print(f"Processing file: {filename}")
    print(f"file {filepath} is exist: {os.path.exists(filepath)}")

    try:
        config_parser = ConfigParser(config)
        converter_cls = config_parser.get_converter_cls()
        converter = converter_cls(
            config=config_parser.generate_config_dict(),
            artifact_dict=model_refs,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service()
        )
        rendered = converter(filepath)

        # 保存结果
        out_folder = config_parser.get_output_folder(filepath)
        save_output(rendered, out_folder, config_parser.get_base_filename(filepath))

        completion_time = time.time()
        time_difference = completion_time - entry_time
        print(f"Time taken to process {filename}: {time_difference}")

        return {
            "filename": filename,
            "markdown": rendered.markdown,
            "markdown_filepath": markdown_filepath,
            # "metadata": rendered.metadata,
            "metadata_filepath": out_meta_filepath,
            "status": "ok",
            "time": time_difference
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {filename}. {e}")


@app.get("/")
def server():
    """
    服务状态检查接口
    """
    return {"message": "Welcome to Marker-api"}


@app.post("/convert")
async def convert_file_to_markdown(
        file: UploadFile = None,
        filepath: str = None,
        output_folder: str = OUTPUT_FOLDER,
        max_pages: int = None,
        start_page: int = None,
        metadata: Optional[dict] = None,
        langs: Optional[List[str]] = None,
        batch_multiplier: int = 1,
        ocr_all_pages: bool = False,
        use_llm: bool = False,
        output_format: str = "markdown",
):
    """
    单文件转换接口
    支持上传文件或指定文件路径
    """
    kwargs = {
        "max_pages": max_pages,
        "start_page": start_page,
        "metadata": metadata,
        "langs": langs,
        "batch_multiplier": batch_multiplier,
        "ocr_all_pages": ocr_all_pages,
        "use_llm": use_llm,
        "output_format": output_format
    }
    if file:
        with tempfile.NamedTemporaryFile('w+b', suffix=".pdf", delete=False) as temp_pdf:
            temp_pdf.write(await file.read())
            temp_pdf.seek(0)
            filepath = temp_pdf.name
    result = convert_single_file(
        filepath=filepath,
        output_folder=output_folder,
        **kwargs
    )
    try:
        if os.path.exists(filepath):
            os.unlink(filepath)
    except Exception as e:
        print(f"Error deleting temporary file {filepath}: {e}")

    return result


def worker_init(model_dict):
    """
    初始化worker进程的模型
    确保在worker进程中使用相同的GPU设备
    """
    global model_refs
    model_refs = model_dict
    
    # 在worker进程中设置CUDA设备为GPU 0
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # 强制使用 GPU 0


def process_single_pdf(args):
    """
    处理单个PDF文件的worker函数
    返回与原接口兼容的结果格式
    """
    filepath, cli_options = args
    entry_time = time.time()

    try:
        config_parser = ConfigParser(cli_options)
        converter_cls = config_parser.get_converter_cls()

        converter = converter_cls(
            config=config_parser.generate_config_dict(),
            artifact_dict=model_refs,  # 直接使用全局model_refs
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service()
        )

        rendered = converter(filepath)
        out_folder = config_parser.get_output_folder(filepath)
        save_output(rendered, out_folder, config_parser.get_base_filename(filepath))

        completion_time = time.time()
        filename = os.path.basename(filepath)
        markdown_filepath = get_markdown_filepath(out_folder, filename)
        out_meta_filepath = markdown_filepath.rsplit(".", 1)[0] + "_meta.json"
        return {
            "filename": filename,
            "markdown": rendered.markdown,
            "markdown_filepath": markdown_filepath,
            # "metadata": rendered.metadata,
            "metadata_filepath": out_meta_filepath,
            "status": "ok",
            "time": completion_time - entry_time
        }

    except Exception as e:
        return {
            "filename": os.path.basename(filepath),
            "status": "error",
            "error": str(e)
        }


@app.post("/batch_convert")
async def convert_files_to_markdown(
        files: List[UploadFile] = None,
        filepaths: List[str] = None,
        output_folder: str = OUTPUT_FOLDER,
        max_pages: int = None,
        start_page: int = None,
        metadata: Optional[dict] = None,
        langs: Optional[List[str]] = None,
        batch_multiplier: int = 1,
        ocr_all_pages: bool = False,
        use_llm: bool = False,
        output_format: str = "markdown",
        workers: int = os.environ.get("WORKERS", 4),
):
    """
    批量文件转换接口
    使用multiprocessing处理多个文件
    """
    print([file.filename for file in files] if files else "no files")
    if not filepaths and not files:
        raise HTTPException(status_code=400, detail="No files provided")

    temp_files = []  # 存储临时文件路径
    results = []
    try:
        # 处理文件上传或文件路径
        if not files:
            for filepath in filepaths:
                if not os.path.exists(filepath):
                    raise HTTPException(status_code=400, detail=f"File not found: {filepath}")
        else:
            filepaths = []
            for file in files:
                with tempfile.NamedTemporaryFile('w+b', suffix=".pdf", delete=False) as temp_pdf:
                    content = await file.read()
                    temp_pdf.write(content)
                    temp_pdf.flush()
                    filepaths.append(temp_pdf.name)
                    temp_files.append(temp_pdf.name)
                await file.close()

        # 准备转换配置
        config = {
            "output_dir": output_folder,
            "max_pages": max_pages,
            "start_page": start_page,
            "metadata": metadata,
            "languages": langs,
            "batch_multiplier": batch_multiplier,
            "force_ocr": ocr_all_pages,
            "use_llm": use_llm,
            "output_format": output_format,
            "disable_multiprocessing": True  # 禁用嵌套多进程
        }

        total_processes = min(len(filepaths), workers)

        # 使用全局model_refs
        global model_refs

        # 使用进程池处理文件
        with mp.Pool(processes=total_processes,
                     initializer=worker_init,
                     initargs=(model_refs,)) as pool:
            results = list(pool.imap(process_single_pdf, [(f, config) for f in filepaths]))

    finally:
        # 清理临时文件
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                print(f"Error deleting temporary file {temp_file}: {e}")

    return results


def main():
    """
    启动服务
    支持配置host、port和workers数量
    """
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the marker-api server.")
    parser.add_argument("--host", default="0.0.0.0", help="Host IP address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")

    args = parser.parse_args()

    # 设置环境变量
    os.environ["WORKERS"] = str(args.workers)

    uvicorn.run("server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
