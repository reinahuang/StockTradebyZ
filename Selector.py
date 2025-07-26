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
    """BBI+KDJ选股器（少妇战法）"""
    
    def __init__(self, 
                 j_threshold: float = 1.0,
                 bbi_min_window: int = 20,
                 max_window: int = 60,
                 price_range_pct: float = 0.5,
                 bbi_q_threshold: float = 0.1,
                 j_q_threshold: float = 0.1):
        self.j_threshold = j_threshold
        self.bbi_min_window = bbi_min_window
        self.max_window = max_window
        self.price_range_pct = price_range_pct
        self.bbi_q_threshold = bbi_q_threshold
        self.j_q_threshold = j_q_threshold
        
        logger.info("BBIKDJSelector初始化完成")
    
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
        """计算BBI指标"""
        if len(df) < 24:
            return pd.Series([np.nan] * len(df), index=df.index)
            
        ma3 = df['close'].rolling(3).mean()
        ma6 = df['close'].rolling(6).mean()
        ma12 = df['close'].rolling(12).mean()
        ma24 = df['close'].rolling(24).mean()
        
        bbi = (ma3 + ma6 + ma12 + ma24) / 4
        return bbi
    
    def select(self, trade_date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> List[str]:
        """执行选股逻辑"""
        picks = []
        
        for code, df in data.items():
            if len(df) < self.max_window:
                continue
                
            try:
                valid_df = df[df["date"] <= trade_date]
                if len(valid_df) < self.bbi_min_window:
                    continue
                
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
                
                # 计算分位数
                j_valid = j.dropna()
                bbi_valid = bbi.dropna()
                
                if len(j_valid) < 10 or len(bbi_valid) < 10:
                    continue
                
                j_quantile = (current_j <= j_valid).mean()
                bbi_quantile = (current_bbi <= bbi_valid).mean()
                
                # 选股条件（OR关系）
                j_condition = (current_j <= self.j_threshold and 
                              j_quantile <= self.j_q_threshold)
                
                bbi_condition = bbi_quantile <= self.bbi_q_threshold
                
                if j_condition and bbi_condition:
                    picks.append(code)
                    logger.debug("%s 选中: J=%.2f(分位%.2f%%), BBI分位%.2f%%",
                               code, current_j, j_quantile*100, bbi_quantile*100)
                
            except Exception as e:
                logger.warning("处理股票 %s 时出错: %s", code, e)
                continue
        
        logger.info("BBIKDJSelector 共选出 %d 只股票", len(picks))
        return sorted(picks)


class BBIShortLongSelector(Selector):
    """BBI短长均线选股器（补票战法）"""
    
    def __init__(self, 
                 n_short: int = 3,
                 n_long: int = 21,
                 m: int = 3,
                 bbi_min_window: int = 24,  # 改为24，确保BBI计算准确
                 max_window: int = 60,
                 bbi_q_threshold: float = 0.2):
        
        self.n_short = n_short
        self.n_long = n_long
        self.m = m
        self.bbi_min_window = bbi_min_window
        self.max_window = max_window
        self.bbi_q_threshold = bbi_q_threshold
        
        logger.info("BBIShortLongSelector初始化完成")
    
    def calculate_bbi(self, df: pd.DataFrame) -> pd.Series:
        """计算BBI指标"""
        if len(df) < 24:
            return pd.Series([np.nan] * len(df), index=df.index)
            
        ma3 = df['close'].rolling(3).mean()
        ma6 = df['close'].rolling(6).mean()
        ma12 = df['close'].rolling(12).mean()
        ma24 = df['close'].rolling(24).mean()
        
        bbi = (ma3 + ma6 + ma12 + ma24) / 4
        return bbi
    
    def select(self, trade_date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> List[str]:
        """补票战法选股逻辑 - 修正版"""
        picks = []
        
        for code, df in data.items():
            if len(df) < self.max_window:
                continue
                
            try:
                valid_df = df[df["date"] <= trade_date]
                if len(valid_df) < max(self.n_long, 24):
                    continue
                
                recent_df = valid_df.tail(self.max_window)
                
                # 计算BBI
                bbi = self.calculate_bbi(recent_df)
                if bbi.isna().all():
                    continue
                
                # 计算短期和长期均线
                ma_short = recent_df['close'].rolling(self.n_short).mean()
                ma_long = recent_df['close'].rolling(self.n_long).mean()
                
                if ma_short.isna().all() or ma_long.isna().all():
                    continue
                
                current_bbi = bbi.iloc[-1]
                current_ma_short = ma_short.iloc[-1]
                current_ma_long = ma_long.iloc[-1]
                current_price = recent_df['close'].iloc[-1]
                
                if pd.isna(current_bbi) or pd.isna(current_ma_short) or pd.isna(current_ma_long):
                    continue
                
                # 计算BBI分位数
                bbi_valid = bbi.dropna()
                if len(bbi_valid) < 10:
                    continue
                
                bbi_quantile = (current_bbi <= bbi_valid).mean()
                
                # 补票战法逻辑修正：
                # 1. 短期均线上穿长期均线（确认上涨趋势）
                ma_condition = current_ma_short > current_ma_long
                
                # 2. BBI已经脱离底部但还未过高（补票时机）
                # 从低分位数(20%)上升到中等分位数(20%-60%)之间
                bbi_breaking_condition = (bbi_quantile > self.bbi_q_threshold and 
                                        bbi_quantile <= 0.6)
                
                # 3. 价格有上涨动能（近期上涨确认）
                if len(recent_df) >= 5:
                    price_momentum = current_price > recent_df['close'].iloc[-5]  # 5日内上涨
                else:
                    price_momentum = True
                
                # 4. 均线距离适中（不是暴涨状态，还有补票机会）
                ma_distance = abs(current_ma_short - current_ma_long) / current_ma_long
                ma_distance_reasonable = ma_distance <= 0.05  # 均线距离不超过5%
                
                # 综合判断：确认上涨趋势 + BBI脱离底部 + 有上涨动能 + 均线距离合理
                if ma_condition and bbi_breaking_condition and price_momentum and ma_distance_reasonable:
                    picks.append(code)
                    logger.debug("%s 选中[补票]: MA短期%.2f > MA长期%.2f, BBI分位%.2f%% (脱离底部), 均线距离%.2f%%",
                               code, current_ma_short, current_ma_long, bbi_quantile*100, ma_distance*100)
                else:
                    logger.debug("%s 未选中: MA条件:%s, BBI脱离条件:%s(%.2f%%), 动能:%s, 距离:%s(%.2f%%)",
                               code, ma_condition, bbi_breaking_condition, bbi_quantile*100, 
                               price_momentum, ma_distance_reasonable, ma_distance*100)
            
            except Exception as e:
                logger.warning("处理股票 %s 时出错: %s", code, e)
                continue
    
        logger.info("BBIShortLongSelector(补票战法) 共选出 %d 只股票", len(picks))
        return sorted(picks)


class BreakoutVolumeKDJSelector(Selector):
    """突破成交量KDJ选股器（TePu战法）"""
    
    def __init__(self, 
                 j_threshold: float = 1,
                 j_q_threshold: float = 0.1,
                 up_threshold: float = 3.0,
                 volume_threshold: float = 0.6667,
                 offset: int = 15,
                 max_window: int = 60,
                 price_range_pct: float = 0.5):
        
        self.j_threshold = j_threshold
        self.j_q_threshold = j_q_threshold
        self.up_threshold = up_threshold
        self.volume_threshold = volume_threshold
        self.offset = offset
        self.max_window = max_window
        self.price_range_pct = price_range_pct
        
        logger.info("BreakoutVolumeKDJSelector初始化完成")
    
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
    
    def select(self, trade_date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> List[str]:
        """TePu战法选股逻辑"""
        picks = []
        
        for code, df in data.items():
            if len(df) < self.max_window:
                continue
                
            try:
                valid_df = df[df["date"] <= trade_date]
                if len(valid_df) < self.offset + 10:
                    continue
                
                recent_df = valid_df.tail(self.max_window)
                
                # 计算KDJ
                k, d, j = self.calculate_kdj(recent_df)
                if j is None or j.isna().all():
                    continue
                
                current_j = j.iloc[-1]
                if pd.isna(current_j):
                    continue
                
                # 计算价格突破
                current_price = recent_df['close'].iloc[-1]
                offset_price = recent_df['close'].iloc[-(self.offset+1)] if len(recent_df) > self.offset else recent_df['close'].iloc[0]
                price_change_pct = (current_price - offset_price) / offset_price * 100
                
                # 计算成交量比率
                recent_volume = recent_df['volume'].tail(5).mean()
                earlier_volume = recent_df['volume'].head(len(recent_df)//2).mean()
                volume_ratio = recent_volume / earlier_volume if earlier_volume > 0 else 0
                
                # 计算J值分位数
                j_valid = j.dropna()
                if len(j_valid) < 10:
                    continue
                
                j_quantile = (current_j <= j_valid).mean()
                
                # 选股条件
                j_condition = (current_j <= self.j_threshold and j_quantile <= self.j_q_threshold)
                breakout_condition = price_change_pct >= self.up_threshold
                volume_condition = volume_ratio >= self.volume_threshold
                
                if j_condition and breakout_condition and volume_condition:
                    picks.append(code)
                    logger.debug("%s 选中: J=%.2f(分位%.2f%%), 涨幅%.2f%%, 量比%.2f",
                               code, current_j, j_quantile*100, price_change_pct, volume_ratio)
                
            except Exception as e:
                logger.warning("处理股票 %s 时出错: %s", code, e)
                continue
        
        logger.info("BreakoutVolumeKDJSelector 共选出 %d 只股票", len(picks))
        return sorted(picks)


class PeakKDJSelector(Selector):
    """峰值KDJ选股器（填坑战法）"""
    
    def __init__(self, 
                 j_threshold: float = 10,
                 max_window: int = 100,
                 fluc_threshold: float = 0.03,
                 j_q_threshold: float = 0.1,
                 gap_threshold: float = 0.2):
        
        self.j_threshold = j_threshold
        self.max_window = max_window
        self.fluc_threshold = fluc_threshold
        self.j_q_threshold = j_q_threshold
        self.gap_threshold = gap_threshold
        
        logger.info("PeakKDJSelector初始化完成")
    
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
    
    def select(self, trade_date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> List[str]:
        """填坑战法选股逻辑"""
        picks = []
        
        for code, df in data.items():
            if len(df) < self.max_window:
                continue
                
            try:
                valid_df = df[df["date"] <= trade_date]
                if len(valid_df) < 30:
                    continue
                
                recent_df = valid_df.tail(self.max_window)
                
                # 计算KDJ
                k, d, j = self.calculate_kdj(recent_df)
                if j is None or j.isna().all():
                    continue
                
                current_j = j.iloc[-1]
                if pd.isna(current_j):
                    continue
                
                # 找到近期高点
                high_window = 20
                if len(recent_df) < high_window:
                    continue
                
                recent_highs = recent_df['high'].tail(high_window)
                peak_price = recent_highs.max()
                current_price = recent_df['close'].iloc[-1]
                
                # 计算回调幅度
                pullback_pct = (peak_price - current_price) / peak_price
                
                # 计算波动率
                returns = recent_df['close'].pct_change().dropna()
                volatility = returns.std() if len(returns) > 1 else 0
                
                # 计算J值分位数
                j_valid = j.dropna()
                if len(j_valid) < 10:
                    continue
                
                j_quantile = (current_j <= j_valid).mean()
                
                # 选股条件
                j_condition = (current_j <= self.j_threshold and j_quantile <= self.j_q_threshold)
                pullback_condition = pullback_pct >= self.gap_threshold
                volatility_condition = volatility <= self.fluc_threshold
                
                if j_condition and pullback_condition and volatility_condition:
                    picks.append(code)
                    logger.debug("%s 选中: J=%.2f(分位%.2f%%), 回调%.2f%%, 波动率%.2f%%",
                               code, current_j, j_quantile*100, pullback_pct*100, volatility*100)
                
            except Exception as e:
                logger.warning("处理股票 %s 时出错: %s", code, e)
                continue
        
        logger.info("PeakKDJSelector 共选出 %d 只股票", len(picks))
        return sorted(picks)


# 别名定义
class 少妇战法(BBIKDJSelector):
    """少妇战法的别名"""
    pass

class 补票战法(BBIShortLongSelector):
    """补票战法的别名"""
    pass

class TePu战法(BreakoutVolumeKDJSelector):
    """TePu战法的别名"""
    pass

class 填坑战法(PeakKDJSelector):
    """填坑战法的别名"""
    pass