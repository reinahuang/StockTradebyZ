from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any

import akshare as ak
import pandas as pd
from tqdm import tqdm

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("fetch_hk.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("fetch_hk")

def load_hk_appendix(file_path: str = "appendix_hk.json") -> List[str]:
    """从港股配置文件加载股票列表"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        codes = config.get("data", [])
        logger.info("从 %s 加载了 %d 只港股", file_path, len(codes))
        return codes
    except FileNotFoundError:
        logger.error("港股配置文件 %s 不存在", file_path)
        return []
    except json.JSONDecodeError as e:
        logger.error("港股配置文件 %s 格式错误: %s", file_path, e)
        return []

def get_hk_market_stocks() -> List[str]:
    """获取港股市场股票列表"""
    try:
        # 获取恒生指数成分股
        logger.info("正在获取恒生指数成分股...")
        df = ak.index_stock_cons_sina(symbol='HSI')
        codes = df['code'].tolist()
        logger.info("获取恒生指数成分股 %d 只", len(codes))
        return codes
    except Exception as e:
        logger.warning("获取恒生指数成分股失败: %s", e)
        try:
            # 备选：获取港股通成分股
            logger.info("尝试获取港股通成分股...")
            df = ak.tool_trade_date_hist_sina()  # 需要替换为实际的港股通接口
            return []
        except Exception as e2:
            logger.warning("获取港股通成分股失败: %s", e2)
            # 使用预定义的知名港股
            backup_codes = [
                '00700', '00939', '00005', '00388', '01299', '02318',
                '00175', '01398', '03988', '02628', '01109', '00883'
            ]
            logger.info("使用备选港股列表 %d 只", len(backup_codes))
            return backup_codes

def validate_hk_data(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """验证港股数据质量"""
    if df.empty:
        return df
    
    # 检查必要列
    required_cols = ["date", "open", "close", "high", "low", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning("港股 %s 缺少列: %s", code, missing_cols)
        return pd.DataFrame()
    
    # 数据类型转换
    try:
        df["date"] = pd.to_datetime(df["date"])
        numeric_cols = ["open", "close", "high", "low", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    except Exception as e:
        logger.warning("港股 %s 数据类型转换失败: %s", code, e)
        return pd.DataFrame()
    
    # 过滤无效数据
    before_len = len(df)
    df = df.dropna(subset=["open", "close", "high", "low"])
    df = df[df["close"] > 0]  # 价格必须大于0
    after_len = len(df)
    
    if before_len != after_len:
        logger.debug("港股 %s 过滤了 %d 条无效记录", code, before_len - after_len)
    
    return df.sort_values("date").reset_index(drop=True)

def fetch_hk_kline(code: str, start: str, end: str, max_retries: int = 3) -> pd.DataFrame:
    """抓取港股K线数据"""
    for attempt in range(1, max_retries + 1):
        try:
            # 随机延迟避免被限制
            delay = random.uniform(0.5, 2.0) * attempt
            time.sleep(delay)
            
            logger.debug("正在抓取港股 %s，第 %d 次尝试", code, attempt)
            
            # 使用AKShare港股接口
            df = ak.stock_hk_hist(
                symbol=code,
                period="daily",
                start_date=start,
                end_date=end,
                adjust="qfq"  # 前复权
            )
            
            if df is None or df.empty:
                logger.debug("港股 %s 返回空数据", code)
                continue
            
            # 标准化列名
            column_map = {
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount"
            }
            
            # 检查并重命名列
            if "日期" in df.columns:
                df = df.rename(columns=column_map)
            
            # 验证数据
            df = validate_hk_data(df, code)
            if not df.empty:
                logger.debug("港股 %s 成功获取 %d 条记录", code, len(df))
                return df[["date", "open", "close", "high", "low", "volume"]].copy()
            
        except Exception as e:
            logger.warning("港股 %s 第 %d 次抓取失败: %s", code, attempt, e)
            if attempt < max_retries:
                backoff_time = random.uniform(2, 8) * attempt
                time.sleep(backoff_time)
    
    logger.error("港股 %s 最终抓取失败", code)
    return pd.DataFrame()

def fetch_and_save_hk(code: str, start: str, end: str, out_dir: Path, incremental: bool = True) -> None:
    """抓取并保存单只港股数据"""
    csv_path = out_dir / f"{code}_hk.csv"
    
    # 增量更新逻辑
    if incremental and csv_path.exists():
        try:
            existing = pd.read_csv(csv_path, parse_dates=["date"])
            if not existing.empty and "date" in existing.columns:
                last_date = existing["date"].max()
                start_dt = pd.to_datetime(start, format="%Y%m%d")
                
                if last_date.date() >= start_dt.date():
                    # 从最后日期开始更新
                    start = (last_date + pd.Timedelta(days=1)).strftime("%Y%m%d")
                    logger.debug("港股 %s 增量更新，从 %s 开始", code, start)
                    
                    end_dt = pd.to_datetime(end, format="%Y%m%d")
                    if last_date.date() >= end_dt.date():
                        logger.debug("港股 %s 已是最新，无需更新", code)
                        return
        except Exception as e:
            logger.warning("读取港股 %s 历史数据失败，将重新下载: %s", code, e)
    
    # 抓取数据
    new_df = fetch_hk_kline(code, start, end)
    if new_df.empty:
        logger.warning("港股 %s 无新数据", code)
        return
    
    # 合并数据（如果是增量更新）
    if csv_path.exists() and incremental:
        try:
            old_df = pd.read_csv(csv_path, parse_dates=["date"])
            combined_df = pd.concat([old_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset="date", keep="last")
            combined_df = combined_df.sort_values("date").reset_index(drop=True)
            new_df = combined_df
            logger.debug("港股 %s 合并历史数据，总计 %d 条记录", code, len(new_df))
        except Exception as e:
            logger.warning("港股 %s 合并数据失败，使用新数据: %s", code, e)
    
    # 保存数据
    try:
        new_df.to_csv(csv_path, index=False)
        logger.info("港股 %s 保存完成，共 %d 条记录", code, len(new_df))
    except Exception as e:
        logger.error("港股 %s 保存失败: %s", code, e)

def main():
    parser = argparse.ArgumentParser(description="抓取港股历史数据")
    parser.add_argument("--tickers", help="指定港股代码，用逗号分隔")
    parser.add_argument("--start", default="20220101", help="开始日期 (YYYYMMDD)")
    parser.add_argument("--end", default="today", help="结束日期 (YYYYMMDD)")
    parser.add_argument("--out", default="./hk_data", help="输出目录")
    parser.add_argument("--workers", type=int, default=3, help="并发线程数")
    parser.add_argument("--appendix-only", action="store_true", help="只抓取 appendix_hk.json 中的股票")
    parser.add_argument("--config", default="appendix_hk.json", help="港股配置文件路径")
    parser.add_argument("--incremental", action="store_true", default=True, help="增量更新模式")
    
    args = parser.parse_args()
    
    # 处理日期
    if args.start == "today":
        start = dt.date.today().strftime("%Y%m%d")
    else:
        start = args.start
        
    if args.end == "today":
        end = dt.date.today().strftime("%Y%m%d")
    else:
        end = args.end
    
    # 创建输出目录
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取股票列表
    if args.tickers:
        codes = [code.strip() for code in args.tickers.split(",") if code.strip()]
        logger.info("使用用户指定的 %d 只港股", len(codes))
    elif args.appendix_only:
        codes = load_hk_appendix(args.config)
        if not codes:
            logger.error("无法从配置文件获取港股列表，程序退出")
            sys.exit(1)
    else:
        codes = get_hk_market_stocks()
        if not codes:
            logger.error("无法获取港股市场列表，程序退出")
            sys.exit(1)
    
    logger.info("开始抓取 %d 只港股，日期范围: %s → %s", len(codes), start, end)
    logger.info("输出目录: %s", out_dir.resolve())
    
    # 多线程抓取
    success_count = 0
    failed_codes = []
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # 提交任务
        futures = {
            executor.submit(fetch_and_save_hk, code, start, end, out_dir, args.incremental): code
            for code in codes
        }
        
        # 处理结果
        for future in tqdm(as_completed(futures), total=len(futures), desc="抓取港股"):
            code = futures[future]
            try:
                future.result()
                success_count += 1
            except Exception as e:
                logger.error("港股 %s 抓取任务失败: %s", code, e)
                failed_codes.append(code)
    
    # 输出统计信息
    logger.info("港股数据抓取完成！")
    logger.info("成功: %d 只，失败: %d 只", success_count, len(failed_codes))
    if failed_codes:
        logger.warning("失败的港股代码: %s", ", ".join(failed_codes))
    logger.info("数据保存至: %s", out_dir.resolve())

if __name__ == "__main__":
    main()