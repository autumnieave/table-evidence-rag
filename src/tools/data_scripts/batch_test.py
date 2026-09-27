"""
batch_test_parallel.py
多进程并行批量测试脚本
"""

import json
import pandas as pd
from pathlib import Path
import time
import multiprocessing as mp
from functools import partial
import os

# 导入 RAG 系统
from core.pipeline import RAGPipeline, RAGConfig


def parse_question_cell(cell_value):
    """解析问题单元格（与原来相同）"""
    if pd.isna(cell_value):
        return []
    try:
        items = json.loads(cell_value)
        if isinstance(items, list):
            return [item.get("Q", "") for item in items if isinstance(item, dict)]
        return []
    except json.JSONDecodeError:
        return [str(cell_value).strip()]


def process_single_row(row_data, max_retries=3):
    """
    处理单行数据的函数，将在子进程中运行。
    每个子进程独立创建 RAGPipeline 实例，避免状态冲突。
    """
    question_id = row_data["编号"]
    question_type = row_data["问题类型"]
    raw_questions = row_data["问题"]
    questions = parse_question_cell(raw_questions)

    if not questions:
        print(f"[PID {os.getpid()}] ⚠️ {question_id} 无有效问题，跳过")
        return None

    print(f"[PID {os.getpid()}] 开始处理 {question_id} ({question_type})")

    # ----- 关键：每个子进程独立初始化配置和 Pipeline -----
    config = RAGConfig()
    config.ENABLE_MULTI_TURN = True

    pipeline = RAGPipeline(config)
    pipeline.agent_mode_enabled = True

    user_id = f"{question_id}"
    # 重置对话状态（确保全新会话）
    pipeline.conversation_state = type(pipeline.conversation_state)()
    pipeline.conversation_state.user_id = user_id

    aggregated_rounds = []
    for q in questions:
        retry_count = 0
        answer = None
        while retry_count < max_retries:
            try:
                answer = pipeline.agent_query(q, user_id=user_id, verbose=False)
                break
            except Exception as e:
                retry_count += 1
                print(f"[PID {os.getpid()}] ⚠️ {question_id} 失败 (尝试 {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    time.sleep(2 ** retry_count)  # 指数退避
                else:
                    answer = {
                        "content": f"处理失败，已重试{max_retries}次: {e}",
                        "image": [],
                        "references": []
                    }

        # 兼容字符串返回（意图澄清）
        if isinstance(answer, str):
            answer = {"content": answer, "image": [], "references": []}

        aggregated_rounds.append(answer)
        time.sleep(1)  # 问题间稍作延迟，避免 API 突发压力

    sql = pipeline.get_accumulated_sql(user_id=user_id)
    rounds_output = [{"Q": q, "A": a} for q, a in zip(questions, aggregated_rounds)]

    # 返回该行的结果字典
    return {
        "编号": question_id,
        "问题类型": question_type,
        "原始问题": raw_questions,
        "SQL语句": sql,
        "结构化输出": json.dumps(rounds_output, ensure_ascii=False)
    }


def run_batch_test_parallel(max_workers=4):
    """多进程批量测试主函数"""
    # 读取问题数据
    df_questions = pd.read_excel(
        "测试数据/20-29-30.xlsx",
        sheet_name="Sheet1"
    )

    # 准备任务列表
    tasks = []
    for _, row in df_questions.iterrows():
        tasks.append({
            "编号": row["编号"],
            "问题类型": row["问题类型"],
            "问题": row["问题"]
        })

    print(f"共 {len(tasks)} 个问题，使用 {max_workers} 个进程并行处理")

    # 使用进程池执行
    with mp.Pool(processes=max_workers) as pool:
        # 使用 functools.partial 传递固定参数（如 max_retries）
        worker_func = partial(process_single_row, max_retries=3)
        results = []
        for i, result in enumerate(pool.imap_unordered(worker_func, tasks)):
            if result is not None:
                results.append(result)
                print(f"进度: {len(results)}/{len(tasks)} 完成")
            else:
                print(f"进度: {i+1}/{len(tasks)} 跳过无效问题")

    # 按编号排序（可选）
    results.sort(key=lambda x: x["编号"])

    # 写入 Excel
    df_result = pd.DataFrame(results)
    df_result.to_excel("result_3_parallel.xlsx", index=False)
    print("\n✅ 并行测试完成，结果已保存到 result_3_parallel.xlsx")


if __name__ == "__main__":
    # 设置启动方法（Windows 下可能需要 'spawn'）
    mp.set_start_method('spawn', force=True)
    run_batch_test_parallel(max_workers=3)  # 根据 API 限制调整并发数