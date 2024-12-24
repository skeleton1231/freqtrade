from typing import Dict

import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.persistence import Trade
from freqtrade.strategy import (
    IntParameter,
    IStrategy,
)


class SampleStrategyEnhanced(IStrategy):
    """
    改进版策略：
      - 在达到 timeout 时，只要有利润，就卖出。
      - ROI 标准依然适用。
    """

    INTERFACE_VERSION = 3

    can_short = False

    # 不使用 trailing_stop
    trailing_stop = False

    # 全局固定止损 (不考虑止损情况)
    stoploss = -0.03

    # ROI 在 config 中 minimal_roi 设定
    timeframe = "5m"
    process_only_new_candles = True

    # 超参: 让用户可 hyperopt
    buy_rsi = IntParameter(low=10, high=50, default=30, space="buy", optimize=True)
    sell_rsi = IntParameter(low=50, high=90, default=70, space="sell", optimize=True)

    # 启动需要的蜡烛数量
    startup_candle_count = 200

    def populate_indicators(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """
        添加指标：
          - RSI
          - Bollinger Bands
          - TEMA
        """
        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # Bollinger Bands
        boll = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lowerband"] = boll["lower"]
        dataframe["bb_middleband"] = boll["mid"]
        dataframe["bb_upperband"] = boll["upper"]

        # TEMA
        dataframe["tema"] = ta.TEMA(dataframe, timeperiod=9)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """
        入场逻辑：
          - RSI 从下往上突破买入阈值 (buy_rsi)
          - TEMA 在布林中轨以下且向上
        """
        dataframe.loc[:, "enter_long"] = 0

        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe["rsi"], self.buy_rsi.value)
                & (dataframe["tema"] < dataframe["bb_middleband"])
                & (dataframe["tema"] > dataframe["tema"].shift(1))  # TEMA 上升
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """
        出场逻辑：
          - 达到标准 ROI 条件
          - RSI 从下往上突破卖出阈值 (sell_rsi)
          - TEMA 在布林中轨以上且向下
        """
        dataframe.loc[:, "exit_long"] = 0

        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe["rsi"], self.sell_rsi.value)
                & (dataframe["tema"] > dataframe["bb_middleband"])
                & (dataframe["tema"] < dataframe["tema"].shift(1))  # TEMA 下降
            ),
            "exit_long",
        ] = 1

        return dataframe

    def custom_stoploss(
        self, pair: str, trade: Trade, current_time, current_rate, current_profit, **kwargs
    ) -> float:
        """
        自定义退出逻辑：
        - 到达 timeout 时，只要有利润就退出。
        """
        # 默认使用配置中的 stoploss 值
        default_stop = self.stoploss  # 默认 -10%

        # 检查 trade 是否达到 timeout
        timeout_reached = (
            current_time - trade.open_date_utc
        ).total_seconds() / 60 >= trade.open_order_timeout
        if timeout_reached and current_profit > 0:
            # 如果超时并且有利润，立即退出
            return 0  # 当前价格触发退出

        # 否则，保持默认止损
        return default_stop
