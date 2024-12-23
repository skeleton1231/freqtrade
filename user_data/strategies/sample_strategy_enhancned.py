# user_data/strategies/SampleStrategyEnhanced.py

from typing import Dict

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.persistence import Trade
from freqtrade.strategy import (
    DecimalParameter,
    IntParameter,
    IStrategy,
    merge_informative_pair,
    stoploss_from_open,
)


class SampleStrategyEnhanced(IStrategy):
    """
    改进版示例策略:
      - 仅做多
      - 适度放宽 RSI 阈值
      - 用 Bollinger + TEMA 参考
      - 可选: 自定义止损 (ATR-based)
    """

    INTERFACE_VERSION = 3

    can_short = False

    # 使用自定义止损?
    use_custom_stoploss = False  # 若想开启ATR止损,就设True并写 custom_stoploss()

    # 全局固定止损(跟 config 保持一致):
    stoploss = -0.10

    # ROI 在 config 里已经设置 minimal_roi

    timeframe = "5m"
    process_only_new_candles = True

    # 不使用 trailing_stop, 如果想启用看config
    trailing_stop = False

    # 超参: 让用户可 hyperopt
    buy_rsi = IntParameter(low=10, high=50, default=30, space="buy", optimize=True)
    sell_rsi = IntParameter(low=50, high=90, default=70, space="sell", optimize=True)

    # startup
    startup_candle_count = 200

    def populate_indicators(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # Bollinger
        boll = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lowerband"] = boll["lower"]
        dataframe["bb_middleband"] = boll["mid"]
        dataframe["bb_upperband"] = boll["upper"]

        # TEMA
        dataframe["tema"] = ta.TEMA(dataframe, timeperiod=9)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """
        入场: RSI 从下往上突破 buy_rsi & TEMA < 中轨 & TEMA上升
        """
        dataframe.loc[:, "enter_long"] = 0

        # RSI crossed above buy_rsi
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe["rsi"], self.buy_rsi.value)
                & (dataframe["tema"] < dataframe["bb_middleband"])
                & (dataframe["tema"] > dataframe["tema"].shift(1))
            ),
            "enter_long"
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """
        出场: RSI crosses above sell_rsi & TEMA>中轨 & TEMA向下
        """
        dataframe.loc[:, "exit_long"] = 0

        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe["rsi"], self.sell_rsi.value)
                & (dataframe["tema"] > dataframe["bb_middleband"])
                & (dataframe["tema"] < dataframe["tema"].shift(1))
            ),
            "exit_long"
        ] = 1

        return dataframe

    # 可选自定义止损
    # def custom_stoploss(self, pair: str, trade: Trade, current_time, current_rate, current_profit, **kwargs) -> float:
    #     """
    #     示例: 当行情波动(ATR)过大时容忍度更高, 以避免被洗; 当已盈利时收紧止损等.
    #     """
    #     default_stop = self.stoploss  # -0.10
    #
    #     # 取当前 dataframe
    #     dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
    #     if dataframe is None or dataframe.empty:
    #         return default_stop
    #
    #     # 例如计算14日ATR
    #     atr = ta.ATR(dataframe, timeperiod=14).iloc[-1]
    #     # 你可拿atr / current_rate来算额外容忍...
    #
    #     # 简单示例, 当盈利>5%时收紧止损到 -3%
    #     if current_profit > 0.05:
    #         return max(default_stop, -0.03)
    #
    #     # 否则用默认-10%
    #     return default_stop
