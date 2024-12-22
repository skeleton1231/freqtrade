import datetime
import logging
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy

# Initialize the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set log level to DEBUG for detailed logs


class FlexibleStrategy(IStrategy):
    """
    Trading strategy with improved risk/reward management,
    dynamic position sizing, and enhanced signal quality.
    """

    INTERFACE_VERSION = 3
    use_custom_stoploss = True

    # Strategy settings
    can_short = False
    informative_timeframes = ["1h"]

    # Parameter optimization
    buy_rsi = IntParameter(20, 50, default=40, optimize=True)
    sell_rsi = IntParameter(50, 80, default=70, optimize=True)
    adx_threshold = IntParameter(15, 35, default=20, optimize=True)
    macd_hist_threshold = DecimalParameter(0.0, 0.5, default=0.2, optimize=True)
    volume_threshold = DecimalParameter(0.5, 2.5, default=1.2, optimize=True)

    def informative_pairs(self):
        """Define informative pairs for multi-timeframe analysis."""
        # 定义主交易对列表
        main_pairs = [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT",
            "XRP/USDT", "DOT/USDT", "MATIC/USDT", "LTC/USDT", "SHIB/USDT",
            "AVAX/USDT", "LINK/USDT", "ATOM/USDT", "UNI/USDT", "FTM/USDT"
        ]
        # 根据每个主交易对生成多时间框架配对
        informative_pairs = [(pair, "1h") for pair in main_pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Populate indicators for the given dataframe."""
        logger.debug(f"Populating indicators for pair: {metadata['pair']}")

        # Core indicators
        dataframe["rsi"] = ta.RSI(dataframe)
        dataframe["adx"] = ta.ADX(dataframe)
        macd = ta.MACD(dataframe)
        dataframe["macd"], dataframe["macdsignal"], dataframe["macdhist"] = (
            macd["macd"],
            macd["macdsignal"],
            macd["macdhist"],
        )

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lowerband"], dataframe["bb_middleband"], dataframe["bb_upperband"] = (
            bollinger["lower"],
            bollinger["mid"],
            bollinger["upper"],
        )

        # Average True Range
        dataframe["atr"] = ta.ATR(dataframe)

        logger.debug("Indicators populated successfully.")
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define entry signals."""
        logger.debug(f"Calculating entry trend for pair: {metadata['pair']}")

        dataframe.loc[
            (
                (dataframe["rsi"] < self.buy_rsi.value)
                & (dataframe["adx"] > self.adx_threshold.value)
                & (dataframe["macdhist"] > self.macd_hist_threshold.value)
                & (
                    dataframe["volume"]
                    > dataframe["volume"].rolling(14).mean() * self.volume_threshold.value
                )
                & (dataframe["close"] < dataframe["bb_lowerband"])
            ),
            "enter_long",
        ] = 1

        logger.debug("Entry signals calculated.")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define exit signals."""
        logger.debug(f"Calculating exit trend for pair: {metadata['pair']}")
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
        """Custom stoploss logic."""
        logger.debug(f"Calculating custom stoploss for pair: {pair}")
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe is not None and not dataframe.empty:
            atr = dataframe["atr"].iloc[-1]
            atr_mean = dataframe["atr"].rolling(20).mean().iloc[-1]
            stoploss = -0.10  # Default stoploss

            if atr > atr_mean * 1.5:
                stoploss = -0.20
            if current_profit > 0.10:
                stoploss = max(stoploss, -atr / current_rate * 1.5)
            elif current_profit < 0.05:
                stoploss = max(stoploss, -atr / current_rate * 2.0)
            logger.debug(f"Stoploss adjusted to {stoploss}")
            return stoploss

        logger.warning(f"No ATR data available for pair: {pair}, using default stoploss.")
        return -0.10

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """Dynamic stake sizing based on market conditions."""
        logger.debug(f"Calculating stake amount for pair: {pair}")
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe is not None and not dataframe.empty:
            rsi_1h = dataframe["rsi"].iloc[-1]
            if rsi_1h > 50:
                logger.debug(f"RSI (1h) is {rsi_1h}. Reducing stake to half.")
                return max_stake * 0.5  # Use half stake in overbought conditions
        return proposed_stake
