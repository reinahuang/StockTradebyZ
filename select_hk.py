from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# ---------- 日志 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("select_hk_results.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("select_hk")

def load_hk_data(data_dir: Path, codes: List[str]) -> Dict[str, pd.DataFrame]:
    """加载港股数据"""
    data: Dict[str, pd.DataFrame] = {}
    
    for code in codes:
        csv_path = data_dir / f"{code}_hk.csv"
        if not csv_path.exists():
            logger.warning("港股数据文件不存在: %s", csv_path)
            continue
            
        try:
            df = pd.read_csv(csv_path, parse_dates=["date"])
            if not df.empty:
                df = df.sort_values("date").reset_index(drop=True)
                data[code] = df
                logger.debug("加载港股 %s: %d 条记录", code, len(df))
        except Exception as e:
            logger.warning("加载港股 %s 失败: %s", code, e)
    
    logger.info("成功加载 %d 只港股数据", len(data))
    return data

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

def load_config(config_path: Path) -> List[Dict[str, Any]]:
    """加载选股器配置（与A股相同的逻辑）"""
    if not config_path.exists():
        logger.error("配置文件 %s 不存在", config_path)
        sys.exit(1)
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    
    if isinstance(cfg, list):
        selectors = cfg
    elif isinstance(cfg, dict) and "selectors" in cfg:
        selectors = cfg["selectors"]
    elif isinstance(cfg, dict):
        selectors = [cfg]
    else:
        logger.error("配置文件格式错误")
        sys.exit(1)
    
    if not selectors:
        logger.error("配置文件中没有选股器配置")
        sys.exit(1)
    
    logger.info("加载了 %d 个选股器配置", len(selectors))
    return selectors

def instantiate_selector(cfg: Dict[str, Any]):
    """实例化选股器（与A股相同的逻辑）"""
    cls_name: str = cfg.get("class")
    if not cls_name:
        raise ValueError("缺少 class 字段")
    
    try:
        module = importlib.import_module("Selector")
        cls = getattr(module, cls_name)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"无法加载 Selector.{cls_name}: {e}")
    
    params = cfg.get("params", {})
    try:
        instance = cls(**params)
    except Exception as e:
        raise RuntimeError(f"实例化 {cls_name} 失败: {e}")
    
    return cfg.get("alias", cls_name), instance

def get_hk_categories(codes: List[str]) -> Dict[str, List[str]]:
    """获取港股分类信息"""
    try:
        with open("appendix_hk.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        categories = config.get("_categories", {})
        descriptions = config.get("_description", {})
        
        result = {}
        for code in codes:
            code_categories = []
            for category, category_codes in categories.items():
                if code in category_codes:
                    code_categories.append(category)
            
            if code in descriptions:
                company_info = descriptions[code]
                result[code] = f"{company_info} ({', '.join(code_categories) if code_categories else '其他'})"
            else:
                result[code] = f"港股{code} ({', '.join(code_categories) if code_categories else '其他'})"
        
        return result
    except Exception as e:
        logger.warning("获取港股分类信息失败: %s", e)
        return {code: f"港股{code}" for code in codes}

def main():
    parser = argparse.ArgumentParser(description="港股选股工具")
    parser.add_argument("--data-dir", default="./hk_data", help="港股数据目录")
    parser.add_argument("--config", default="./configs_hk.json", help="港股选股器配置文件")
    parser.add_argument("--date", help="指定交易日期 (YYYY-MM-DD)")
    parser.add_argument("--tickers", help="指定港股代码，用逗号分隔")
    parser.add_argument("--appendix-only", action="store_true", help="只处理 appendix_hk.json 中的港股")
    parser.add_argument("--hk-config", default="appendix_hk.json", help="港股股票池配置文件")

    args = parser.parse_args()

    # 检查数据目录
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error("港股数据目录不存在: %s", data_dir)
        sys.exit(1)

    # 确定股票池
    if args.tickers:
        codes = [code.strip() for code in args.tickers.split(",") if code.strip()]
        logger.info("使用用户指定的 %d 只港股", len(codes))
    elif args.appendix_only:
        codes = load_hk_appendix(args.hk_config)
        if not codes:
            logger.error("无法从配置文件获取港股列表")
            sys.exit(1)
    else:
        # 默认使用所有 _hk.csv 文件
        hk_files = list(data_dir.glob("*_hk.csv"))
        codes = [f.stem.replace("_hk", "") for f in hk_files]
        logger.info("从数据目录发现 %d 只港股", len(codes))

    if not codes:
        logger.error("没有找到任何港股代码")
        sys.exit(1)

    # 加载数据
    logger.info("开始加载港股数据...")
    data = load_hk_data(data_dir, codes)
    if not data:
        logger.error("没有成功加载任何港股数据")
        sys.exit(1)

    # 确定交易日期
    trade_date = (
        pd.to_datetime(args.date)
        if args.date
        else max(df["date"].max() for df in data.values())
    )
    if not args.date:
        logger.info("未指定 --date，使用最近日期 %s", trade_date.date())

    # 加载选股器配置
    logger.info("加载选股器配置...")
    selector_cfgs = load_config(Path(args.config))

    # 获取港股分类信息
    hk_categories = get_hk_categories(list(data.keys()))

    # 逐个选股器运行
    total_picks = set()
    for cfg in selector_cfgs:
        if cfg.get("activate", True) is False:
            logger.info("跳过未激活的选股器: %s", cfg.get("alias", cfg.get("class")))
            continue

        try:
            alias, selector = instantiate_selector(cfg)
        except Exception as e:
            logger.error("跳过配置 %s：%s", cfg, e)
            continue

        logger.info("运行选股器: %s", alias)
        picks = selector.select(trade_date, data)

        # 输出结果
        logger.info("")
        logger.info("============== 港股选股结果 [%s] ==============", alias)
        logger.info("选股日期: %s", trade_date.date())
        logger.info("符合条件港股数: %d", len(picks))
        
        if picks:
            logger.info("选中的港股:")
            for i, code in enumerate(picks, 1):
                category_info = hk_categories.get(code, f"港股{code}")
                logger.info("  %2d. %s - %s", i, code, category_info)
        else:
            logger.info("  无符合条件的港股")
        
        logger.info("=" * 60)
        total_picks.update(picks)

    # 汇总结果
    logger.info("")
    logger.info("================ 港股选股汇总 ================")
    logger.info("总计选中港股数: %d", len(total_picks))
    if total_picks:
        logger.info("汇总选中港股:")
        for i, code in enumerate(sorted(total_picks), 1):
            category_info = hk_categories.get(code, f"港股{code}")
            logger.info("  %2d. %s - %s", i, code, category_info)
    logger.info("=" * 60)

if __name__ == "__main__":
    main()