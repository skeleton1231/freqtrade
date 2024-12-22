import datetime
import logging
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # DEBUG日志，便于分析


class FlexibleStrategy(IStrategy):
    """
    示例策略（已移除volume过滤）：
    - 以Hyperopt参考参数构建
    - 不使用成交量条件（volume 过滤）
    - 保留自定义止损逻辑 + 全局 stoploss=-0.345
    """

    # ----------------------------
    # 基本配置
    # ----------------------------
    INTERFACE_VERSION = 3

    # 只做多
    can_short = False

    # 使用自定义止损
    use_custom_stoploss = True

    # 全局固定止损（来自超参结果）
    stoploss = -0.345

    # 多时间框架（可按需删除）
    informative_timeframes = ["1h"]

    # ----------------------------
    # 可调参区（示例）
    # 已去除 volume_threshold
    # ----------------------------
    adx_threshold = IntParameter(10, 35, default=32, optimize=True, space="buy")
    buy_rsi       = IntParameter(20, 60, default=37, optimize=True, space="buy")
    macd_hist_threshold = DecimalParameter(0.0, 0.5, default=0.4, optimize=True, space="buy")

    # 仍可保留卖出 RSI
    sell_rsi = IntParameter(50, 80, default=74, optimize=True, space="sell")

    # ----------------------------
    # 多时间框架对
    # ----------------------------
    def informative_pairs(self):
        """
        若需要多时间框架，定义在这里。
        如果不想用多TF，可删除本方法 + informative_timeframes。
        """
        main_pairs = [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT",
            "XRP/USDT", "DOT/USDT", "MATIC/USDT", "LTC/USDT", "SHIB/USDT",
            "AVAX/USDT", "LINK/USDT", "ATOM/USDT", "UNI/USDT", "FTM/USDT"
        ]
        # 获取这些对的1h级别数据
        return [(pair, "1h") for pair in main_pairs]

    # ----------------------------
    # 指标计算
    # ----------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        logger.debug(f"Populating indicators for pair: {metadata['pair']}")
        # RSI & ADX
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

        # MACD(12,26,9)
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

        # 布林带
        boll = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe),
            window=20,
            stds=2
        )
        dataframe["bb_lowerband"] = boll["lower"]
        dataframe["bb_middleband"] = boll["mid"]
        dataframe["bb_upperband"] = boll["upper"]

        # ATR
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        logger.debug("Indicators populated successfully.")
        return dataframe

    # ----------------------------
    # 买入逻辑（无volume过滤）
    # ----------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        logger.debug(f"Calculating entry trend for pair: {metadata['pair']}")
        dataframe.loc[:, "enter_long"] = 0

        dataframe.loc[
            (
                (dataframe["rsi"] < self.buy_rsi.value)
                & (dataframe["adx"] > self.adx_threshold.value)
                & (dataframe["macdhist"] > self.macd_hist_threshold.value)
                & (dataframe["close"] < dataframe["bb_middleband"])
            ),
            "enter_long"
        ] = 1

        logger.debug("Entry signals calculated.")
        return dataframe

    # ----------------------------
    # 卖出逻辑
    # ----------------------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        logger.debug(f"Calculating exit trend for pair: {metadata['pair']}")
        dataframe.loc[:, "exit_long"] = 0

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

    # ----------------------------
    # 自定义止损 (可选)
    # ----------------------------
    def custom_stoploss(
        self,
        pair: str,
        trade,
        current_time: datetime.datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        logger.debug(f"Calculating custom stoploss for pair: {pair}")
        stoploss = self.stoploss  # 默认值 -0.345

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe is not None and not dataframe.empty:
            atr = dataframe["atr"].iloc[-1]
            atr_mean = dataframe["atr"].rolling(20).mean().iloc[-1]

            # 若波动率远高于均值 => -0.20
            if atr > atr_mean * 1.5:
                stoploss = -0.20  # 相对 -0.345 更小绝对值(=-20%)，=>更宽松

            # 如果盈利>10% => 收紧
            if current_profit > 0.10:
                # 计算一个基于ATR的止损
                stop = - (atr / current_rate * 1.5)
                stoploss = max(stoploss, stop)

            # 如果盈利<5% => 继续宽松
            elif current_profit < 0.05:
                stop = - (atr / current_rate * 2.0)
                stoploss = max(stoploss, stop)

            logger.debug(f"Stoploss adjusted to {stoploss}")
        else:
            logger.warning(f"No ATR data for {pair}, using default stoploss {stoploss}.")
        return stoploss

    # ----------------------------
    # 动态仓位管理 (可选)
    # ----------------------------
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
        若1h RSI>50，减半仓位
        """
        logger.debug(f"Calculating stake amount for pair: {pair}")
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe="1h")

        if dataframe is not None and not dataframe.empty:
            rsi_1h = dataframe["rsi"].iloc[-1]
            if rsi_1h > 50:
                logger.debug(f"RSI(1h)={rsi_1h}, reducing stake to half.")
                return max_stake * 0.5

        return proposed_stake
