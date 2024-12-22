import datetime
import logging
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set log level to DEBUG for detailed logs


class FlexibleStrategy(IStrategy):
    """
    优化后示例策略（已移除成交量相关过滤）：
    - 放宽买入条件（RSI, MACD阈值等）
    - 提高开单频率
    - 自定义止损逻辑（可选）
    """

    INTERFACE_VERSION = 3
    use_custom_stoploss = True

    # 允许做多（现货）配置
    can_short = False

    # 使用多时间框架（如不需要，可删）
    informative_timeframes = ["1h"]

    # 参数区（供优化/调参）
    buy_rsi = IntParameter(20, 60, default=45, optimize=True, space="buy")
    sell_rsi = IntParameter(50, 80, default=70, optimize=True, space="sell")
    adx_threshold = IntParameter(10, 35, default=20, optimize=True, space="buy")
    macd_hist_threshold = DecimalParameter(0.0, 0.5, default=0.1, optimize=True, space="buy")
    # volume_threshold 保留在此，但不再用于买入条件
    volume_threshold = DecimalParameter(0.5, 2.0, default=1.0, optimize=True, space="buy")

    def informative_pairs(self):
        """
        定义多时间框架所需的交易对（如不需要，可删）。
        """
        main_pairs = [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT",
            "XRP/USDT", "DOT/USDT", "MATIC/USDT", "LTC/USDT", "SHIB/USDT",
            "AVAX/USDT", "LINK/USDT", "ATOM/USDT", "UNI/USDT", "FTM/USDT"
        ]
        informative_pairs = [(pair, "1h") for pair in main_pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        为给定的 dataframe 计算各种技术指标。
        """
        logger.debug(f"Populating indicators for pair: {metadata['pair']}")

        # 1) RSI & ADX
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

        # 2) MACD
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

        # 3) 布林带
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe),
            window=20,
            stds=2
        )
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]

        # 4) ATR
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        logger.debug("Indicators populated successfully.")
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        买入逻辑（无成交量过滤）。
        """
        logger.debug(f"Calculating entry trend for pair: {metadata['pair']}")

        dataframe.loc[:, "enter_long"] = 0

        # 策略示例：RSI < buy_rsi + ADX > threshold + MACD hist > threshold + 收盘价低于布林中轨
        dataframe.loc[
            (
                (dataframe["rsi"] < self.buy_rsi.value)
                & (dataframe["adx"] > self.adx_threshold.value)
                & (dataframe["macdhist"] > self.macd_hist_threshold.value)
                & (dataframe["close"] < dataframe["bb_middleband"])
                # 如果想强调下轨，可换成:
                # & (dataframe["close"] < dataframe["bb_lowerband"] * 1.02)
            ),
            "enter_long"
        ] = 1

        logger.debug("Entry signals calculated.")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        卖出逻辑。
        """
        logger.debug(f"Calculating exit trend for pair: {metadata['pair']}")
        dataframe.loc[:, "exit_long"] = 0

        # 示例：RSI > sell_rsi + MACD直方图<0 + 价格>中轨
        dataframe.loc[
            (
                (dataframe["rsi"] > self.sell_rsi.value)
                & (dataframe["macdhist"] < 0)
                & (dataframe["close"] > dataframe["bb_middleband"])
            ),
            "exit_long"
        ] = 1

        logger.debug("Exit signals calculated.")
        return dataframe

    def custom_stoploss(
        self, pair: str, trade, current_time, current_rate, current_profit, **kwargs
    ) -> float:
        """
        自定义止损逻辑示例：
        - 若ATR高于均值，则加大止损
        - 当前盈利超10%自动收紧止损
        - 当前盈利<5%时则更宽容
        """
        logger.debug(f"Calculating custom stoploss for pair: {pair}")

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        stoploss = -0.10  # 默认

        if dataframe is not None and not dataframe.empty:
            atr = dataframe["atr"].iloc[-1]
            atr_mean = dataframe["atr"].rolling(20).mean().iloc[-1]

            if atr > atr_mean * 1.5:
                stoploss = -0.20

            if current_profit > 0.10:
                stop = - (atr / current_rate * 1.5)
                stoploss = max(stoploss, stop)
            elif current_profit < 0.05:
                stop = - (atr / current_rate * 2.0)
                stoploss = max(stoploss, stop)

            logger.debug(f"Stoploss adjusted to {stoploss}")
        else:
            logger.warning(f"No ATR data available for pair: {pair}, using default stoploss.")
        return stoploss

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime.datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """
        动态仓位管理示例：参考1h的RSI。
        当1h RSI>50时，减半仓位；否则用默认仓位。
        """
        logger.debug(f"Calculating stake amount for pair: {pair}")
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)

        if dataframe is not None and not dataframe.empty:
            rsi_1h = dataframe["rsi"].iloc[-1]
            if rsi_1h > 50:
                logger.debug(f"RSI (1h) is {rsi_1h}. Reducing stake to half.")
                return max_stake * 0.5

        return proposed_stake
