from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

# ---------- 日志 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        # 将日志写入文件
        logging.FileHandler("select_results.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("select")


# ---------- 工具 ----------

def load_data(data_dir: Path, codes: Iterable[str]) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for code in codes:
        fp = data_dir / f"{code}.csv"
        if not fp.exists():
            logger.warning("%s 不存在，跳过", fp.name)
            continue
        df = pd.read_csv(fp, parse_dates=["date"]).sort_values("date")
        frames[code] = df
    return frames


def load_config(cfg_path: Path) -> List[Dict[str, Any]]:
    if not cfg_path.exists():
        logger.error("配置文件 %s 不存在", cfg_path)
        sys.exit(1)
    with cfg_path.open(encoding="utf-8") as f:
        cfg_raw = json.load(f)

    # 兼容三种结构：单对象、对象数组、或带 selectors 键
    if isinstance(cfg_raw, list):
        cfgs = cfg_raw
    elif isinstance(cfg_raw, dict) and "selectors" in cfg_raw:
        cfgs = cfg_raw["selectors"]
    else:
        cfgs = [cfg_raw]

    if not cfgs:
        logger.error("configs.json 未定义任何 Selector")
        sys.exit(1)

    return cfgs


def load_appendix_codes() -> List[str]:
    """从 appendix.json 加载股票代码列表"""
    appendix_path = Path("appendix.json")
    if not appendix_path.exists():
        logger.warning("appendix.json 文件不存在，返回空列表")
        return []
    
    try:
        with appendix_path.open(encoding="utf-8") as f:
            data = json.load(f)
        codes = data.get("data", [])
        logger.info("从 appendix.json 加载了 %d 只股票", len(codes))
        return codes
    except (json.JSONDecodeError, KeyError) as e:
        logger.error("读取 appendix.json 失败: %s", e)
        return []


def load_hs300_codes() -> List[str]:
    """获取沪深300成分股列表"""
    try:
        import akshare as ak
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


def load_a500_codes() -> List[str]:
    """获取 A500 (中证500) 成分股列表"""
    try:
        import akshare as ak
        logger.info("正在获取 A500 (中证500) 成分股列表...")
        
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
        logger.error("获取 A500 成分股失败: %s", e)
        # 返回一些中证500的代表性股票作为备选
        backup_codes = ['002027', '002129', '002142', '002252', '002311', '002375',
                       '002405', '002493', '002508', '002624', '002677', '002709']
        logger.info("使用备选A500股票列表 %d 只", len(backup_codes))
        return backup_codes


def load_combined_codes() -> List[str]:
    """加载沪深300、A500和appendix.json中的股票，去重后返回"""
    hs300_codes = load_hs300_codes()
    a500_codes = load_a500_codes()
    appendix_codes = load_appendix_codes()
    
    # 合并并去重
    combined_codes = list(set(hs300_codes + a500_codes + appendix_codes))
    logger.info("沪深300 + A500 + appendix.json 总计 %d 只股票 (沪深300: %d, A500: %d, appendix: %d)", 
                len(combined_codes), len(hs300_codes), len(a500_codes), len(appendix_codes))
    
    return combined_codes


def instantiate_selector(cfg: Dict[str, Any]):
    """动态加载 Selector 类并实例化"""
    cls_name: str = cfg.get("class")
    if not cls_name:
        raise ValueError("缺少 class 字段")

    try:
        module = importlib.import_module("Selector")
        cls = getattr(module, cls_name)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"无法加载 Selector.{cls_name}: {e}") from e

    params = cfg.get("params", {})
    return cfg.get("alias", cls_name), cls(**params)


# ---------- 主函数 ----------

def main():
    p = argparse.ArgumentParser(description="Run selectors defined in configs.json")
    p.add_argument("--data-dir", default="./data", help="CSV 行情目录")
    p.add_argument("--config", default="./configs.json", help="Selector 配置文件")
    p.add_argument("--date", help="交易日 YYYY-MM-DD；缺省=数据最新日期")
    p.add_argument("--tickers", default="all", help="'all' 或逗号分隔股票代码列表")
    p.add_argument("--only-appendix", action="store_true", help="只处理 appendix.json 中的股票")
    p.add_argument("--combined-appendix", action="store_true", help="只处理沪深300、A500和appendix.json中的股票")
    args = p.parse_args()

    # --- 加载行情 ---
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error("数据目录 %s 不存在", data_dir)
        sys.exit(1)

    codes = (
        [f.stem for f in data_dir.glob("*.csv")]
        if args.tickers.lower() == "all" and not args.only_appendix and not args.combined_appendix
        else [c.strip() for c in args.tickers.split(",") if c.strip()]
        if not args.only_appendix and not args.combined_appendix
        else load_appendix_codes()
        if args.only_appendix
        else load_combined_codes()
    )
    if not codes:
        logger.error("股票池为空！")
        sys.exit(1)

    data = load_data(data_dir, codes)
    if not data:
        logger.error("未能加载任何行情数据")
        sys.exit(1)

    trade_date = (
        pd.to_datetime(args.date)
        if args.date
        else max(df["date"].max() for df in data.values())
    )
    if not args.date:
        logger.info("未指定 --date，使用最近日期 %s", trade_date.date())

    # --- 加载 Selector 配置 ---
    selector_cfgs = load_config(Path(args.config))

    # --- 逐个 Selector 运行 ---
    for cfg in selector_cfgs:
        if cfg.get("activate", True) is False:
            continue
        try:
            alias, selector = instantiate_selector(cfg)
        except Exception as e:
            logger.error("跳过配置 %s：%s", cfg, e)
            continue

        picks = selector.select(trade_date, data)

        # 将结果写入日志，同时输出到控制台
        logger.info("")
        logger.info("============== 选股结果 [%s] ==============", alias)
        logger.info("交易日: %s", trade_date.date())
        logger.info("符合条件股票数: %d", len(picks))
        logger.info("%s", ", ".join(picks) if picks else "无符合条件股票")


if __name__ == "__main__":
    main()
