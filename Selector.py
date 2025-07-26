from __future__ import annotations

import logging
from typing import Dict, List
import pandas as pd
import numpy as np

logger = logging.getLogger("select")


class Selector:
    """选股器基类"""
    
    def select(self, trade_date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> List[str]:
        """
        选股主方法
        
        Args:
            trade_date: 交易日期
            data: 股票数据字典 {股票代码: DataFrame}
            
        Returns:
            符合条件的股票代码列表
        """
        raise NotImplementedError


class BBIKDJSelector(Selector):
    """
    BBI+KDJ选股器（少妇战法）
    
    基于BBI（多空指标）和KDJ指标的组合选股策略
    核心思路：寻找技术指标处于相对低位，有反弹潜力的股票
    """
    
    def __init__(self, 
                 j_threshold: float = 1.0,          # J值绝对阈值
                 bbi_min_window: int = 20,          # BBI最小窗口
                 max_window: int = 60,              # 最大计算窗口
                 price_range_pct: float = 0.5,      # 价格位置百分比
                 bbi_q_threshold: float = 0.1,      # BBI分位数阈值
                 j_q_threshold: float = 0.1):       # J值分位数阈值
        """
        初始化BBIKDJSelector参数
        """
        self.j_threshold = j_threshold
        self.bbi_min_window = bbi_min_window
        self.max_window = max_window
        self.price_range_pct = price_range_pct
        self.bbi_q_threshold = bbi_q_threshold
        self.j_q_threshold = j_q_threshold
        
        logger.info("BBIKDJSelector初始化完成，参数: j_threshold=%.2f, j_q_threshold=%.2f, bbi_q_threshold=%.2f", 
                   self.j_threshold, self.j_q_threshold, self.bbi_q_threshold)
    
    def calculate_kdj(self, df: pd.DataFrame, n: int = 9) -> tuple:
        """计算KDJ指标"""
        if len(df) < n:
            return None, None, None
            
        low_n = df['low'].rolling(n).min()
        high_n = df['high'].rolling(n).max()
        rsv = ((df['close'] - low_n) / (high_n - low_n)) * 100
        k = rsv.ewm(com=2).mean()
        d = k.ewm(com=2).mean()
        j = 3 * k - 2 * d
        return k, d, j
    
    def calculate_bbi(self, df: pd.DataFrame) -> pd.Series:
        """计算BBI指标（多空指标）"""
        if len(df) < 24:  # 至少需要24天数据
            return pd.Series([np.nan] * len(df), index=df.index)
            
        ma3 = df['close'].rolling(3).mean()
        ma6 = df['close'].rolling(6).mean()
        ma12 = df['close'].rolling(12).mean()
        ma24 = df['close'].rolling(24).mean()
        
        bbi = (ma3 + ma6 + ma12 + ma24) / 4
        return bbi
    
    def select(self, trade_date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> List[str]:
        """
        执行BBI+KDJ选股
        
        选股逻辑（OR关系 + 分位数优先）：
        1. J值条件：当前J值 <= j_threshold OR J值分位数 <= j_q_threshold
        2. BBI条件：BBI分位数 <= bbi_q_threshold  
        3. 价格位置：当前价格在近期价格区间的相对低位
        4. 数据质量：确保有足够的历史数据
        """
        picks = []
        
        for code, df in data.items():
            if len(df) < self.max_window:
                continue
                
            try:
                # 获取指定交易日及之前的数据
                valid_df = df[df["date"] <= trade_date]
                if len(valid_df) < self.bbi_min_window:
                    continue
                
                # 获取最近的计算窗口数据
                recent_df = valid_df.tail(self.max_window)
                
                # 计算KDJ指标
                k, d, j = self.calculate_kdj(recent_df)
                if j is None or j.isna().all():
                    continue
                    
                current_j = j.iloc[-1]
                if pd.isna(current_j):
                    continue
                
                # 计算BBI指标
                bbi = self.calculate_bbi(recent_df)
                if bbi.isna().all():
                    continue
                    
                current_bbi = bbi.iloc[-1]
                if pd.isna(current_bbi):
                    continue
                
                # 计算价格位置
                recent_prices = recent_df['close'].tail(self.bbi_min_window)
                if len(recent_prices) < 5:
                    continue
                    
                price_min = recent_prices.min()
                price_max = recent_prices.max()
                current_price = recent_prices.iloc[-1]
                
                if price_max > price_min:
                    price_position = (current_price - price_min) / (price_max - price_min)
                else:
                    price_position = 0.5
                
                # 计算分位数
                j_valid = j.dropna()
                bbi_valid = bbi.dropna()
                
                if len(j_valid) < 10 or len(bbi_valid) < 10:
                    continue
                
                j_quantile = (current_j <= j_valid).mean()
                bbi_quantile = (current_bbi <= bbi_valid).mean()
                
                # 选股条件判断
                # 1. J值条件（OR关系：绝对值或分位数满足一个即可）
                j_condition = (current_j <= self.j_threshold or 
                              j_quantile <= self.j_q_threshold)
                
                # 2. BBI条件（分位数条件）
                bbi_condition = bbi_quantile <= self.bbi_q_threshold
                
                # 3. 价格位置条件
                price_condition = price_position <= self.price_range_pct
                
                # 综合判断（需要同时满足多个条件）
                if j_condition and bbi_condition and price_condition:
                    picks.append(code)
                    
                    logger.debug("%s 选中: J=%.2f(分位%.2f%%), BBI分位%.2f%%, 价格位置%.2f%%",
                               code, current_j, j_quantile*100, bbi_quantile*100, price_position*100)
                else:
                    logger.debug("%s 未选中: J=%.2f(分位%.2f%%, 条件:%s), BBI分位%.2f%%(条件:%s), 价格位置%.2f%%(条件:%s)",
                               code, current_j, j_quantile*100, j_condition, 
                               bbi_quantile*100, bbi_condition, price_position*100, price_condition)
                
            except Exception as e:
                logger.warning("处理股票 %s 时出错: %s", code, e)
                continue
        
        logger.info("BBIKDJSelector 共选出 %d 只股票", len(picks))
        return sorted(picks)


class 少妇战法(BBIKDJSelector):
    """少妇战法的别名，实际使用BBIKDJSelector"""
    pass


class 基础筛选(Selector):
    """基础筛选器示例"""
    
    def __init__(self, min_price: float = 5.0, max_price: float = 100.0):
        self.min_price = min_price
        self.max_price = max_price
    
    def select(self, trade_date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> List[str]:
        picks = []
        for code, df in data.items():
            valid_df = df[df["date"] <= trade_date]
            if len(valid_df) > 0:
                current_price = valid_df["close"].iloc[-1]
                if self.min_price <= current_price <= self.max_price:
                    picks.append(code)
        return sorted(picks)