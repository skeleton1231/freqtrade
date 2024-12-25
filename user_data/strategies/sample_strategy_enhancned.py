from typing import Dict
from pandas import DataFrame
import logging
import talib.abstract as ta
from freqtrade.strategy import IStrategy, IntParameter
from freqtrade.persistence import Trade
from technical import qtpylib


class SampleStrategyEnhanced(IStrategy):
    """
    Enhanced strategy:
    - Exit immediately if timeout is reached and the trend is unfavorable.
    - ROI standards still apply.
    - Stay in trades with favorable trends even after timeout.
    """

    INTERFACE_VERSION = 3

    can_short = False

    # Disable trailing stop
    trailing_stop = False

    # Global stoploss
    stoploss = -0.03  # Default stoploss

    # ROI defined in the config as minimal_roi
    timeframe = "5m"
    process_only_new_candles = True
    use_custom_stoploss = True

    # Hyperparameters for optimization
    buy_rsi = IntParameter(low=10, high=50, default=30, space="buy", optimize=True)
    sell_rsi = IntParameter(low=50, high=90, default=70, space="sell", optimize=True)

    # Required number of candles before strategy starts
    startup_candle_count = 200

    def __init__(self, config: dict) -> None:
        """
        Initialize the strategy with proper logging configuration.
        """
        super().__init__(config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")  # Logger tied to class name
        self.logger.setLevel(logging.DEBUG)  # Set default log level for this strategy

    def populate_indicators(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """
        Add indicators:
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

        self.logger.debug(f"Indicators populated for metadata: {metadata}")
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """
        Entry logic:
        - RSI crosses above the buy threshold (buy_rsi).
        - TEMA below Bollinger middle band and trending upward.
        """
        dataframe.loc[:, "enter_long"] = 0

        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe["rsi"], self.buy_rsi.value)
                & (dataframe["tema"] < dataframe["bb_middleband"])
                & (dataframe["tema"] > dataframe["tema"].shift(1))  # TEMA upward
            ),
            "enter_long",
        ] = 1

        self.logger.debug(f"Entry trend populated for metadata: {metadata}")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """
        Exit logic:
        - Meets ROI standards.
        - RSI crosses above the sell threshold (sell_rsi).
        - TEMA above Bollinger middle band and trending downward.
        """
        dataframe.loc[:, "exit_long"] = 0

        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe["rsi"], self.sell_rsi.value)
                & (dataframe["tema"] > dataframe["bb_middleband"])
                & (dataframe["tema"] < dataframe["tema"].shift(1))  # TEMA downward
            ),
            "exit_long",
        ] = 1

        self.logger.debug(f"Exit trend populated for metadata: {metadata}")
        return dataframe

    def custom_stoploss(
        self, pair: str, trade: Trade, current_time, current_rate, current_profit, **kwargs
    ) -> float:
        """
        Custom exit logic:
        - Exit immediately if timeout is reached and the trend is unfavorable.
        - Stay in position if the trend is favorable, even after timeout.
        """
        default_stop = self.stoploss
        timeout_minutes = 60  # Timeout duration in minutes
        timeout_reached = (current_time - trade.open_date_utc).total_seconds() / 60 >= timeout_minutes

        # Log general stoploss evaluation details
        self.logger.info(
            f"[Custom Stoploss] Pair: {pair} | Timeout: {timeout_reached} | "
            f"Profit: {current_profit:.4f} | Current Rate: {current_rate:.4f} | Opened Since: {trade.open_date_utc}"
        )

        # If timeout is not reached, do not exit
        if not timeout_reached:
            self.logger.debug(f"[Custom Stoploss] Pair: {pair} | Timeout not reached. Holding position.")
            return default_stop

        # Check market trend (e.g., 1-hour EMA trend)
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe="1h")
        if dataframe is not None and not dataframe.empty:
            ema_trend = dataframe["close"].iloc[-1] > dataframe["ema50"].iloc[-1]
            self.logger.info(
                f"[Custom Stoploss] Pair: {pair} | EMA Trend: {'Favorable' if ema_trend else 'Unfavorable'}"
            )

            # If timeout is reached and the trend is unfavorable, exit immediately with any profit
            if current_profit > 0 and not ema_trend:
                self.logger.warning(
                    f"[Custom Stoploss] Exiting Pair: {pair} | Unfavorable Trend | Profit: {current_profit:.4f}"
                )
                return 0  # Exit immediately

        # Timeout reached: Exit with minimum profit if no favorable trend
        if current_profit >= 0.005:  # At least 0.5% profit
            self.logger.warning(
                f"[Custom Stoploss] Exiting Pair: {pair} | Timeout Reached | Profit: {current_profit:.4f}"
            )
            return 0  # Exit immediately

        # Default stoploss behavior
        self.logger.debug(f"[Custom Stoploss] Pair: {pair} | Holding Position | Profit: {current_profit:.4f}")
        return default_stop
