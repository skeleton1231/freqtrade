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
    优化后示例策略：
    - 放宽买入条件（RSI, MACD阈值等）
    - 提高开单频率
    - 加强自定义止损逻辑
    """

    INTERFACE_VERSION = 3
    use_custom_stoploss = True

    # 允许做多（现货）配置
    can_short = False

    # 使用多时间框架
    informative_timeframes = ["1h"]

    # 参数区（供优化）
    buy_rsi = IntParameter(20, 60, default=45, optimize=True, space="buy")
    sell_rsi = IntParameter(50, 80, default=70, optimize=True, space="sell")
    adx_threshold = IntParameter(10, 35, default=20, optimize=True, space="buy")
    macd_hist_threshold = DecimalParameter(0.0, 0.5, default=0.1, optimize=True, space="buy")
    volume_threshold = DecimalParameter(0.5, 2.0, default=1.0, optimize=True, space="buy")

    def informative_pairs(self):
        """
        定义多时间框架所需的交易对。
        注意：Binance US 上若部分交易对不活跃，需要自行检查。
        """
        main_pairs = [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT",
            "XRP/USDT", "DOT/USDT", "MATIC/USDT", "LTC/USDT", "SHIB/USDT",
            "AVAX/USDT", "LINK/USDT", "ATOM/USDT", "UNI/USDT", "FTM/USDT"
        ]
        # 为这些主要交易对都加上1小时级别的数据
        informative_pairs = [(pair, "1h") for pair in main_pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        为给定的 dataframe 计算各种技术指标。
        """

        logger.debug(f"Populating indicators for pair: {metadata['pair']}")

        # 1) 核心指标
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

        # 2) 布林带
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), 
            window=20, 
            stds=2
        )
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]

        # 3) 平均真实波动ATR
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        logger.debug("Indicators populated successfully.")
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        买入逻辑。略微放宽条件，让策略更可能进场。
        """

        logger.debug(f"Calculating entry trend for pair: {metadata['pair']}")

        # 先全部置空
        dataframe.loc[:, "enter_long"] = 0

        # 策略：RSI低于阈值 + ADX高于阈值 + MACD直方图>阈值 + 成交量>滚动均量*volume_threshold
        #       + 收盘价低于布林中轨(或者略贴近下轨)
        # 目的：更有机会捕捉超卖反弹或短期回调
        dataframe.loc[
            (
                (dataframe["rsi"] < self.buy_rsi.value) 
                & (dataframe["adx"] > self.adx_threshold.value)
                & (dataframe["macdhist"] > self.macd_hist_threshold.value)
                & (
                    dataframe["volume"] 
                    > dataframe["volume"].rolling(14).mean() * self.volume_threshold.value
                )
                & (dataframe["close"] < dataframe["bb_middleband"])  
                # 如果你想依旧强调下轨，可以再换成：
                # & (dataframe["close"] < dataframe["bb_lowerband"] * 1.02)
            ),
            "enter_long"
        ] = 1

        logger.debug("Entry signals calculated.")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        卖出逻辑。可根据需求进行灵活调整。
        """

        logger.debug(f"Calculating exit trend for pair: {metadata['pair']}")
        # 先全部置空
        dataframe.loc[:, "exit_long"] = 0

        # 简单示例：RSI 大于卖出阈值 + MACD直方图 < 0 + 当前价高于布林中轨
        dataframe.loc[
            (
                (dataframe["rsi"] > self.sell_rsi.value)
                & (dataframe["macdhist"] < 0)
                & (dataframe["close"] > dataframe["bb_middleband"])
            ),
            "exit_long",
        ] = 1

        logger.debug("Exit signals calculated.")
        return dataframe

    def custom_stoploss(
        self, pair: str, trade, current_time, current_rate, current_profit, **kwargs
    ) -> float:
        """
        自定义止损逻辑示例：
        - 根据当前 K 线的 ATR 与 rolling ATR 均值做动态调整。
        - 若盈利达到一定阈值，收紧止损；亏损时尝试扩大保护。
        """

        logger.debug(f"Calculating custom stoploss for pair: {pair}")

        # 拿到策略当前 dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)

        # 默认为 -0.10
        stoploss = -0.10

        if dataframe is not None and not dataframe.empty:
            atr = dataframe["atr"].iloc[-1]
            atr_mean = dataframe["atr"].rolling(20).mean().iloc[-1]

            # 如果波动率（当前 ATR）显著高于平均水平，则加大容忍度
            if atr > atr_mean * 1.5:
                stoploss = -0.20  # 加大止损空间

            # 如果当前收益已经 >= 10%
            if current_profit > 0.10:
                # 以ATR 为参考值，进一步收紧止损
                # 例如：以ATR为基础动态止盈锁定
                stop = - (atr / current_rate * 1.5)
                stoploss = max(stoploss, stop)

            # 如果当前收益仍较低 (< 5%)
            elif current_profit < 0.05:
                # 以ATR为基础稍微扩大止损
                stop = - (atr / current_rate * 2.0)
                stoploss = max(stoploss, stop)

            logger.debug(f"Stoploss adjusted to {stoploss}")

        else:
            # 如果没拿到数据，就使用默认止损
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
        动态仓位管理示例：参考 1h RSI，若过高则减仓。
        """

        logger.debug(f"Calculating stake amount for pair: {pair}")

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe is not None and not dataframe.empty:
            rsi_1h = dataframe["rsi"].iloc[-1]
            if rsi_1h > 50:
                logger.debug(f"RSI (1h) is {rsi_1h}. Reducing stake to half.")
                return max_stake * 0.5  # 如果1h级别RSI过高，就减仓
        return proposed_stake
