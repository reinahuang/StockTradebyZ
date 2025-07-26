from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import random
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import akshare as ak
import pandas as pd
import tushare as ts
from mootdx.quotes import Quotes
from tqdm import tqdm

warnings.filterwarnings("ignore")

# --------------------------- 全局日志配置 --------------------------- #
LOG_FILE = Path("fetch.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("fetch_mktcap")

# 屏蔽第三方库多余 INFO 日志
for noisy in ("httpx", "urllib3", "_client", "akshare"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# --------------------------- 市值快照 --------------------------- #

def _get_mktcap_ak() -> pd.DataFrame:
    """实时快照，返回列：code, mktcap（单位：元）"""
    for attempt in range(1, 4):
        try:
            df = ak.stock_zh_a_spot_em()
            break
        except Exception as e:
            logger.warning("AKShare 获取市值快照失败(%d/3): %s", attempt, e)
            time.sleep(backoff := random.uniform(1, 3) * attempt)
    else:
        logger.warning("AKShare 连续三次拉取市值快照失败，将使用空的市值数据")
        return pd.DataFrame(columns=["code", "mktcap"])

    df = df[["代码", "总市值"]].rename(columns={"代码": "code", "总市值": "mktcap"})
    df["mktcap"] = pd.to_numeric(df["mktcap"], errors="coerce")
    return df

# --------------------------- 股票池筛选 --------------------------- #

def _get_hs300_codes() -> List[str]:
    """获取沪深300成分股列表"""
    try:
        logger.info("正在获取沪深300成分股列表...")
        # 尝试不同的API方法
        try:
            df = ak.index_stock_cons_sina(symbol='000300')
            codes = df['code'].tolist()
            logger.info("成功获取沪深300成分股 %d 只", len(codes))
            return codes
        except Exception as e1:
            logger.warning("index_stock_cons_sina失败: %s", e1)
            try:
                df = ak.index_stock_cons_csindex(symbol='000300')
                codes = df['成分券代码'].tolist()
                logger.info("成功获取沪深300成分股 %d 只", len(codes))
                return codes
            except Exception as e2:
                logger.warning("index_stock_cons_csindex失败: %s", e2)
                # 作为备选，返回一些知名的沪深300股票
                codes = ['000001', '000002', '000858', '600000', '600036', '600519', 
                        '000568', '002415', '300014', '600276', '000725', '002714',
                        '600887', '002236', '600031', '000063', '600104', '000166']
                logger.info("使用备选沪深300股票列表 %d 只", len(codes))
                return codes
    except Exception as e:
        logger.error("获取沪深300成分股失败: %s", e)
        # 返回一些知名的沪深300股票作为备选
        backup_codes = ['000001', '000002', '000858', '600000', '600036', '600519', 
                       '000568', '002415', '300014', '600276', '000725', '002714']
        logger.info("使用备选沪深300股票列表 %d 只", len(backup_codes))
        return backup_codes


def _get_a500_codes() -> List[str]:
    """获取A500(中证500)成分股列表"""
    try:
        logger.info("正在获取A500(中证500)成分股列表...")
        # 尝试不同的API方法
        try:
            df = ak.index_stock_cons_sina(symbol='000905')
            codes = df['code'].tolist()
            logger.info("成功获取A500成分股 %d 只", len(codes))
            return codes
        except Exception as e1:
            logger.warning("index_stock_cons_sina失败: %s", e1)
            try:
                df = ak.index_stock_cons_csindex(symbol='000905')
                codes = df['成分券代码'].tolist()
                logger.info("成功获取A500成分股 %d 只", len(codes))
                return codes
            except Exception as e2:
                logger.warning("index_stock_cons_csindex失败: %s", e2)
                # 作为备选，返回一些中证500的代表性股票
                codes = ['002027', '002129', '002142', '002252', '002311', '002375',
                        '002405', '002493', '002508', '002624', '002677', '002709',
                        '300033', '300059', '300070', '300122', '300144', '300207']
                logger.info("使用备选A500股票列表 %d 只", len(codes))
                return codes
    except Exception as e:
        logger.error("获取A500成分股失败: %s", e)
        # 返回一些中证500的代表性股票作为备选
        backup_codes = ['002027', '002129', '002142', '002252', '002311', '002375',
                       '002405', '002493', '002508', '002624', '002677', '002709']
        logger.info("使用备选A500股票列表 %d 只", len(backup_codes))
        return backup_codes

def get_constituents(
    min_cap: float,
    max_cap: float,
    exclude_gem: bool,
    include_kcb: bool,
    only_appendix: bool,
    combined_appendix: bool = False,
    mktcap_df: Optional[pd.DataFrame] = None,
) -> List[str]:
    # 附加股票池 appendix.json
    try:
        with open("appendix.json", "r", encoding="utf-8") as f:
            appendix_codes = json.load(f)["data"]
    except FileNotFoundError:
        appendix_codes = []
    
    # 如果只处理 appendix.json 中的股票
    if only_appendix:
        if not appendix_codes:
            logger.warning("appendix.json 为空或不存在，没有股票可处理")
            return []
        logger.info("只处理 appendix.json 中的 %d 只股票", len(appendix_codes))
        return [code.zfill(6) for code in appendix_codes]  # 移动到这里
    
    # 如果使用沪深300 + A500 + appendix.json组合
    if combined_appendix:
        hs300_codes = _get_hs300_codes()
        a500_codes = _get_a500_codes()
        
        # 合并并去重
        combined_codes = list(set(hs300_codes + a500_codes + appendix_codes))
        logger.info("沪深300 + A500 + appendix.json 总计 %d 只股票 (沪深300: %d, A500: %d, appendix: %d)", 
                    len(combined_codes), len(hs300_codes), len(a500_codes), len(appendix_codes))
        return combined_codes

    df = mktcap_df if mktcap_df is not None else _get_mktcap_ak()

    # 如果市值数据为空，返回一个基本的股票列表
    if df.empty:
        logger.warning("市值数据为空，将使用默认股票列表")
        # 返回一些常见的大盘股作为默认
        default_codes = ['000001', '000002', '600000', '600036', '600519', '000858']
        logger.info("使用默认股票列表，共 %d 只股票", len(default_codes))
        return default_codes

    cond = (df["mktcap"] >= min_cap) & (df["mktcap"] <= max_cap)
    
    # 排除创业板、北交所等
    if exclude_gem:
        cond &= ~df["code"].str.startswith(("300", "301", "8", "4"))
    
    # 如果不专门包含科创板，则排除科创板
    if not include_kcb:
        cond &= ~df["code"].str.startswith("688")
    
    codes = df.loc[cond, "code"].str.zfill(6).tolist()
    codes = list(dict.fromkeys(appendix_codes + codes))  # 去重保持顺序

    logger.info("筛选得到 %d 只股票", len(codes))
    return codes

# --------------------------- 历史 K 线抓取 --------------------------- #
COLUMN_MAP_HIST_AK = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "换手率": "turnover",
}

_FREQ_MAP = {
    0: "5m",
    1: "15m",
    2: "30m",
    3: "1h",
    4: "day",
    5: "week",
    6: "mon",
    7: "1m",
    8: "1m",
    9: "day",
    10: "3mon",
    11: "year",
}

# ---------- Tushare 工具函数 ---------- #

def _to_ts_code(code: str) -> str:
    return f"{code.zfill(6)}.SH" if code.startswith(("60", "68", "9")) else f"{code.zfill(6)}.SZ"


def _get_kline_tushare(code: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    ts_code = _to_ts_code(code)
    adj_flag = None if adjust == "" else adjust
    for attempt in range(1, 4):
        try:
            # 使用全局 pro API，如果不存在则重新初始化
            if 'pro' not in globals():
                pro = ts.pro_api()
            df = pro.daily(
                ts_code=ts_code,
                adj=adj_flag,
                start_date=start,
                end_date=end,
            )
            break
        except Exception as e:
            logger.warning("Tushare 拉取 %s 失败(%d/3): %s", code, attempt, e)
            time.sleep(random.uniform(1, 2) * attempt)
    else:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(columns={"trade_date": "date", "vol": "volume"})[
        ["date", "open", "close", "high", "low", "volume"]
    ].copy()
    df["date"] = pd.to_datetime(df["date"])
    df[[c for c in df.columns if c != "date"]] = df[[c for c in df.columns if c != "date"]].apply(
        pd.to_numeric, errors="coerce"
    )    
    return df.sort_values("date").reset_index(drop=True)

# ---------- AKShare 工具函数 ---------- #

def _get_kline_akshare(code: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    for attempt in range(1, 6):  # 增加到5次重试
        try:
            # 根据重试次数动态调整延迟
            delay = random.uniform(0.5, 1.5) * attempt
            time.sleep(delay)
            
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start,
                end_date=end,
                adjust=adjust,
            )
            
            if df is not None and not df.empty:
                # 确保列名正确
                if "日期" in df.columns:
                    df = df.rename(columns=COLUMN_MAP_HIST_AK)
                
                # 检查必要的列是否存在
                required_cols = ["date", "open", "close", "high", "low", "volume"]
                if all(col in df.columns for col in required_cols):
                    # 确保date列是datetime类型
                    df["date"] = pd.to_datetime(df["date"])
                    # 确保数值列是数值类型
                    numeric_cols = ["open", "close", "high", "low", "volume"]
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors="coerce")
                    
                    return df[required_cols].copy()
                else:
                    logger.warning("AKShare返回数据缺少必要列，当前列: %s", list(df.columns))
                    
        except Exception as e:
            logger.warning("AKShare 拉取 %s 失败(%d/5): %s", code, attempt, e)
            # 指数退避，越往后等待时间越长
            backoff_time = random.uniform(2, 8) * (attempt ** 2)
            time.sleep(backoff_time)
    
    logger.error("AKShare 拉取 %s 最终失败，已尝试5次", code)
    return pd.DataFrame()

# ---------- Mootdx 工具函数 ----------

def _get_kline_mootdx(code: str, start: str, end: str, adjust: str, freq_code: int) -> pd.DataFrame:    
    symbol = code.zfill(6)
    freq = _FREQ_MAP.get(freq_code, "day")
    client = Quotes.factory(market="std")
    try:
        df = client.bars(symbol=symbol, frequency=freq, adjust=adjust or None)
    except Exception as e:
        logger.warning("Mootdx 拉取 %s 失败: %s", code, e)
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    
    df = df.rename(
        columns={"datetime": "date", "open": "open", "high": "high", "low": "low", "close": "close", "vol": "volume"}
    )
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    start_ts = pd.to_datetime(start, format="%Y%m%d")
    end_ts = pd.to_datetime(end, format="%Y%m%d")
    df = df[(df["date"].dt.date >= start_ts.date()) & (df["date"].dt.date <= end_ts.date())].copy()    
    df = df.sort_values("date").reset_index(drop=True)    
    return df[["date", "open", "close", "high", "low", "volume"]]

# ---------- 通用接口 ---------- #

def get_kline(
    code: str,
    start: str,
    end: str,
    adjust: str,
    datasource: str,
    freq_code: int = 4,
) -> pd.DataFrame:
    if datasource == "tushare":
        return _get_kline_tushare(code, start, end, adjust)
    elif datasource == "akshare":
        return _get_kline_akshare(code, start, end, adjust)
    elif datasource == "mootdx":        
        return _get_kline_mootdx(code, start, end, adjust, freq_code)
    else:
        raise ValueError("datasource 仅支持 'tushare', 'akshare' 或 'mootdx'")

# ---------- 数据校验 ---------- #

def validate(df: pd.DataFrame) -> pd.DataFrame:
    # 检查数据框是否为空
    if df.empty:
        return df
    
    # 检查是否存在必要的列
    required_columns = ['date', 'open', 'close', 'high', 'low', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.warning("数据缺少必要列: %s，当前列: %s", missing_columns, list(df.columns))
        # 如果缺少关键列，返回空DataFrame
        if 'date' in missing_columns:
            logger.error("数据缺少date列，无法处理")
            return pd.DataFrame(columns=required_columns)
    
    # 去重并排序
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    
    # 检查日期列
    if df["date"].isna().any():
        logger.warning("存在缺失日期，将删除这些行")
        df = df.dropna(subset=['date'])
    
    # 检查未来日期
    if (df["date"] > pd.Timestamp.today()).any():
        logger.warning("数据包含未来日期，将删除这些行")
        df = df[df["date"] <= pd.Timestamp.today()]
    
    return df

def drop_dup_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated()]

# ---------- 单只股票抓取 ---------- #
def fetch_one(
    code: str,
    start: str,
    end: str,
    out_dir: Path,
    incremental: bool,
    datasource: str,
    freq_code: int,
):    
    csv_path = out_dir / f"{code}.csv"

    # 增量更新：若本地已有数据则从最后一天开始
    if incremental and csv_path.exists():
        try:
            existing = pd.read_csv(csv_path, parse_dates=["date"])
            if not existing.empty and "date" in existing.columns:
                last_date = existing["date"].max()
                if last_date.date() > pd.to_datetime(end, format="%Y%m%d").date():
                    logger.debug("%s 已是最新，无需更新", code)
                    return
                start = last_date.strftime("%Y%m%d")
        except Exception as e:
            logger.warning("读取 %s 失败，将重新下载: %s", csv_path, e)

    for attempt in range(1, 4):
        try:            
            new_df = get_kline(code, start, end, "qfq", datasource, freq_code)
            if new_df.empty:
                logger.debug("%s 无新数据", code)
                break
                
            # 验证数据
            new_df = validate(new_df)
            if new_df.empty:
                logger.warning("%s 验证后数据为空", code)
                break
                
            if csv_path.exists() and incremental:
                try:
                    old_df = pd.read_csv(
                        csv_path,
                        parse_dates=["date"],
                        index_col=False
                    )
                    old_df = drop_dup_columns(old_df)
                    new_df = drop_dup_columns(new_df)
                    new_df = (
                        pd.concat([old_df, new_df], ignore_index=True)
                        .drop_duplicates(subset="date")
                        .sort_values("date")
                    )
                except Exception as e:
                    logger.warning("%s 合并数据失败，使用新数据: %s", code, e)
                    
            new_df.to_csv(csv_path, index=False)
            logger.debug("%s 下载完成，共 %d 条记录", code, len(new_df))
            break
            
        except Exception as e:
            logger.warning("%s 第 %d 次抓取失败: %s", code, attempt, e)
            time.sleep(random.uniform(1, 3) * attempt)  # 指数退避
    else:
        logger.error("%s 三次抓取均失败，已跳过！", code)


# ---------- 主入口 ---------- #

def main():
    parser = argparse.ArgumentParser(description="按市值筛选 A 股并抓取历史 K 线")
    parser.add_argument("--datasource", choices=["tushare", "akshare", "mootdx"], default="akshare", help="历史 K 线数据源")
    parser.add_argument("--frequency", type=int, choices=list(_FREQ_MAP.keys()), default=4, help="K线频率编码，参见说明")
    parser.add_argument("--exclude-gem", action="store_true", help="排除创业板/科创板/北交所")
    parser.add_argument("--include-kcb", action="store_true", help="包含科创板（688开头）")
    parser.add_argument("--min-mktcap", type=float, default=5e9, help="最小总市值（含），单位：元")
    parser.add_argument("--max-mktcap", type=float, default=float("+inf"), help="最大总市值（含），单位：元，默认无限制")
    parser.add_argument("--skip-mktcap", action="store_true", help="跳过市值筛选，直接使用本地已有股票")
    parser.add_argument("--only-appendix", action="store_true", help="只处理 appendix.json 中的股票")
    parser.add_argument("--combined-appendix", action="store_true", help="只处理沪深300、A500和appendix.json中的股票")
    parser.add_argument("--tickers", help="指定股票代码，用逗号分隔（如：000001,600000,300001）")
    parser.add_argument("--start", default="20190101", help="起始日期 YYYYMMDD 或 'today'")
    parser.add_argument("--end", default="today", help="结束日期 YYYYMMDD 或 'today'")
    parser.add_argument("--out", default="./data", help="输出目录")
    parser.add_argument("--workers", type=int, default=10, help="并发线程数")
    args = parser.parse_args()

    # ---------- Token 处理 ---------- #
    pro = None
    if args.datasource == "tushare":
        ts_token = " "  # 在这里补充token
        if not ts_token.strip():
            logger.warning("Tushare token 未设置，可能会有访问限制")
        ts.set_token(ts_token)
        try:
            pro = ts.pro_api()
        except Exception as e:
            logger.error("Tushare API 初始化失败: %s", e)
            sys.exit(1)

    # ---------- 日期解析 ---------- #
    start = dt.date.today().strftime("%Y%m%d") if args.start.lower() == "today" else args.start
    end = dt.date.today().strftime("%Y%m%d") if args.end.lower() == "today" else args.end

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 市值快照 & 股票池 ---------- #
    if args.tickers:
        # 使用用户指定的股票代码
        codes_from_filter = [code.strip() for code in args.tickers.split(",") if code.strip()]
        logger.info("使用用户指定的 %d 只股票: %s", len(codes_from_filter), ", ".join(codes_from_filter))
    elif args.only_appendix:
        logger.info("只处理 appendix.json 中的股票")
        mktcap_df = pd.DataFrame(columns=["code", "mktcap"])
        codes_from_filter = get_constituents(
            0,  # min_cap 不重要
            float("+inf"),  # max_cap 不重要
            False,  # exclude_gem 不重要
            False,  # include_kcb 不重要
            True,   # only_appendix = True
            False,  # combined_appendix = False
            mktcap_df=mktcap_df,
        )
    elif args.combined_appendix:
        logger.info("只处理沪深300、A500和appendix.json中的股票")
        mktcap_df = pd.DataFrame(columns=["code", "mktcap"])
        codes_from_filter = get_constituents(
            0,  # min_cap 不重要
            float("+inf"),  # max_cap 不重要
            False,  # exclude_gem 不重要
            False,  # include_kcb 不重要
            False,  # only_appendix = False
            True,   # combined_appendix = True
            mktcap_df=mktcap_df,
        )
    elif args.skip_mktcap:
        logger.info("跳过市值筛选，仅使用本地已有股票")
        mktcap_df = pd.DataFrame(columns=["code", "mktcap"])
        codes_from_filter = []
    else:
        mktcap_df = _get_mktcap_ak()    
        codes_from_filter = get_constituents(
            args.min_mktcap,
            args.max_mktcap,
            args.exclude_gem,
            args.include_kcb,
            False,  # only_appendix = False
            False,  # combined_appendix = False
            mktcap_df=mktcap_df,
        )    
    
    # 加上本地已有的股票，确保旧数据也能更新（除非是只处理特定股票池模式）
    if args.tickers or args.only_appendix or args.combined_appendix:
        codes = codes_from_filter  # 只使用指定的股票池
    else:
        local_codes = [p.stem for p in out_dir.glob("*.csv")]
        codes = sorted(set(codes_from_filter) | set(local_codes))

    if not codes:
        if args.tickers:
            logger.error("指定的股票代码为空或无效！")
        elif args.only_appendix:
            logger.error("appendix.json 为空或不存在，没有股票可处理！")
        elif args.combined_appendix:
            logger.error("沪深300、A500和appendix.json股票池为空，没有股票可处理！")
        else:
            logger.error("筛选结果为空，请调整参数或检查 %s 目录是否有已存在的 CSV 文件！", out_dir)
        sys.exit(1)

    logger.info(
        "开始抓取 %d 支股票 | 数据源:%s | 频率:%s | 日期:%s → %s",
        len(codes),
        args.datasource,
        _FREQ_MAP[args.frequency],
        start,
        end,
    )

    # ---------- 多线程抓取 ---------- #
    # 降低并发数减少网络错误
    max_workers = min(args.workers, 10) if args.datasource == "akshare" else args.workers
    logger.info("使用 %d 个并发线程", max_workers)
    
    completed_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                fetch_one,
                code,
                start,
                end,
                out_dir,
                True,
                args.datasource,
                args.frequency,
            )
            for code in codes
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="下载进度"):
            try:
                future.result()  # 获取结果，这样可以捕获异常
                completed_count += 1
            except Exception as e:
                logger.error("任务执行失败: %s", e)
                failed_count += 1

    logger.info("任务完成: 成功 %d, 失败 %d", completed_count, failed_count)
    logger.info("全部任务完成，数据已保存至 %s", out_dir.resolve())


if __name__ == "__main__":
    main()
