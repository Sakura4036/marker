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

import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import numpy as np

from marker.config.parser import ConfigParser
from marker.models import create_model_dict
from marker.output import get_markdown_filepath, save_output
from marker.settings import settings

# 全局变量
OUTPUT_FOLDER = 'output'
app_data = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    服务启动时加载模型
    支持多GPU配置
    """
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        raise RuntimeError("Set start method to spawn twice. Please try running it again.")

    global app_data
    if settings.TORCH_DEVICE == "mps":
        print("Cannot use MPS with torch multiprocessing share_memory. Using CPU instead.")
        app_data = {}
    else:
        app_data["models"] = create_model_dict()
        for model_name, model in app_data["models"].items():
            if model is not None:
                model.share_memory()
    yield


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
        use_llm: bool = False
) -> Dict:
    """
    转换单个PDF文件
    """
    if not filepath:
        raise HTTPException(status_code=400, detail="No file provided")

    # 配置转换参数
    config = {
        "output_dir": output_folder,
        "max_pages": max_pages,
        "start_page": start_page,
        "metadata": metadata,
        "languages": langs,
        "batch_multiplier": batch_multiplier,
        "force_ocr": ocr_all_pages,
        "use_llm": use_llm
    }

    filename = os.path.basename(filepath)
    markdown_filepath = get_markdown_filepath(output_folder, filename)

    # 检查是否已存在转换结果
    if os.path.exists(markdown_filepath):
        markdown_text = open(markdown_filepath, "r", encoding='utf-8').read()
        out_meta_filepath = markdown_filepath.rsplit(".", 1)[0] + "_meta.json"
        metadata = json.load(open(out_meta_filepath, "r")) if os.path.exists(out_meta_filepath) else {}

        return {
            "filename": filename,
            "markdown": markdown_text,
            "markdown_filepath": markdown_filepath,
            "metadata": metadata,
            "metadata_filepath": out_meta_filepath,
            "status": "ok",
            "time": 0
        }

    # 执行转换
    entry_time = time.time()
    print(f"Processing file: {filename}")

    try:
        config_parser = ConfigParser(config)
        converter_cls = config_parser.get_converter_cls()
        converter = converter_cls(
            config=config_parser.generate_config_dict(),
            artifact_dict=app_data["models"],
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service()
        )
        rendered = converter(filepath)

        # 保存结果
        out_folder = config_parser.get_output_folder(filepath)
        # save_markdown(out_folder, filename, rendered.markdown, rendered.images, rendered.metadata)
        save_output(rendered, out_folder, config_parser.get_base_filename(filepath))

        completion_time = time.time()
        time_difference = completion_time - entry_time
        print(f"Time taken to process {filename}: {time_difference}")

        return {
            "filename": filename,
            "markdown": rendered.markdown,
            "markdown_filepath": markdown_filepath,
            "metadata": rendered.metadata,
            "status": "ok",
            "time": time_difference
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {filename}. \n{e}")


@app.get("/")
def server():
    """
    服务状态检查接口
    """
    return {"message": "Welcome to Marker-api"}


@app.post("/convert")
def convert_file_to_markdown(
        file: UploadFile = None,
        filepath: str = None,
        output_folder: str = None,
        max_pages: int = None,
        start_page: int = None,
        metadata: Optional[dict] = None,
        langs: Optional[List[str]] = None,
        batch_multiplier: int = 1,
        ocr_all_pages: bool = False,
        use_llm: bool = False
):
    """
    单文件转换接口
    支持上传文件或指定文件路径
    """
    if file:
        with tempfile.NamedTemporaryFile('w+b', suffix=".pdf") as temp_pdf:
            temp_pdf.write(file.read())
            temp_pdf.seek(0)
            filepath = temp_pdf.name

    return convert_single_file(
        filepath=filepath,
        output_folder=output_folder,
        max_pages=max_pages,
        start_page=start_page,
        metadata=metadata,
        langs=langs,
        batch_multiplier=batch_multiplier,
        ocr_all_pages=ocr_all_pages,
        use_llm=use_llm
    )


def process_batch(task_args, device_id, workers):
    """
    在指定GPU设备上处理一批文件
    
    Args:
        task_args: 包含文件路径和配置的任务参数列表
        device_id: GPU设备ID
        workers: 每个GPU的worker数量
    """
    # 设置当前进程使用的GPU设备
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    try:
        # 初始化模型并共享内存
        models = create_model_dict()
        for model in models.values():
            if model is not None:
                model.share_memory()

        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for filepath, config in task_args:
                future = executor.submit(
                    convert_single_file,
                    filepath=filepath,
                    **config
                )
                futures.append(future)

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        "status": "error",
                        "error": str(e)
                    })

        return results

    except Exception as e:
        return [{
            "status": "error",
            "error": f"GPU {device_id} batch processing error: {str(e)}"
        }]


@app.post("/batch_convert")
async def convert_files_to_markdown(
        files: List[UploadFile] = None,
        filepaths: List[str] = None,
        output_folder: str = None,
        workers: int = 4,
        num_gpus: int = 1,
        use_llm: bool = False
):
    """
    批量文件转换接口
    支持多workers和多GPU配置
    """
    if not filepaths and not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # 准备文件路径列表
    if not files:
        files_to_process = [(filepath, None) for filepath in filepaths]
    else:
        # 保存上传的文件到临时目录
        files_to_process = []
        for file in files:
            with tempfile.NamedTemporaryFile('w+b', suffix=".pdf", delete=False) as temp_pdf:
                temp_pdf.write(await file.read())
                files_to_process.append((temp_pdf.name, file.filename))

    print(f"Processing {len(files_to_process)} files with {workers} workers on {num_gpus} GPUs")

    # 根据GPU数量拆分任务
    num_gpus = min(num_gpus, len(files_to_process))  # 不要使用多于文件数的GPU
    chunks = np.array_split(files_to_process, num_gpus)

    # 准备每个GPU的任务参数
    tasks = []
    workers_per_gpu = max(1, workers // num_gpus)

    for gpu_id, chunk in enumerate(chunks):
        task_args = []
        for filepath, original_name in chunk:
            config = {
                "output_folder": output_folder,
                "use_llm": use_llm
            }
            task_args.append((filepath, config))
        tasks.append((task_args, gpu_id, workers_per_gpu))

    # 使用进程池并行处理各个GPU上的任务
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    results = []
    with mp.Pool(processes=num_gpus) as pool:
        batch_results = pool.starmap(process_batch, tasks)
        for batch in batch_results:
            results.extend(batch)

    # 清理临时文件
    if files:
        for filepath, _ in files_to_process:
            try:
                os.unlink(filepath)
            except:
                pass

    return results


def main():
    """
    启动服务
    支持配置host、port和GPU数量
    """
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the marker-api server.")
    parser.add_argument("--host", default="0.0.0.0", help="Host IP address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")

    args = parser.parse_args()

    # 设置环境变量
    os.environ["NUM_GPUS"] = str(args.num_gpus)
    os.environ["WORKERS"] = str(args.workers)

    uvicorn.run("server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
