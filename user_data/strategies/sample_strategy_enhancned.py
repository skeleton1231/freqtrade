from typing import Dict
from pandas import DataFrame
import logging
import talib.abstract as ta
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
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

    # Declare additional timeframes
    additional_timeframes = ["1h"]

    # Hyperparameters for optimization
    buy_rsi = IntParameter(low=10, high=50, default=30, space="buy", optimize=True)
    sell_rsi = IntParameter(low=50, high=90, default=70, space="sell", optimize=True)

    # Parameterized timeout and minimum profit
    timeout_minutes = IntParameter(low=30, high=120, default=60, space="custom", optimize=True)
    min_profit = DecimalParameter(
        low=0.002, high=0.01, decimals=4, default=0.005, space="custom", optimize=True
    )

    # Required number of candles before strategy starts
    startup_candle_count = 200

    def __init__(self, config: dict) -> None:
        """
        Initialize the strategy with proper logging configuration.
        """
        super().__init__(config)
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )  # Logger tied to class name
        self.logger.setLevel(logging.DEBUG)  # Set default log level for this strategy

    def populate_indicators(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """
        Add indicators:
        - RSI
        - Bollinger Bands
        - TEMA
        - EMA50 (for main timeframe)
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

        # EMA50 for main timeframe
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)

        self.logger.debug(f"Indicators populated for main timeframe: {metadata}")
        return dataframe

    def populate_indicators_1h(self, dataframe: DataFrame, metadata: Dict) -> DataFrame:
        """
        Add indicators for the 1-hour timeframe:
        - EMA50
        """
        # EMA50 for 1h timeframe
        dataframe["ema50_1h"] = ta.EMA(dataframe, timeperiod=50)

        self.logger.debug(f"Indicators populated for 1h timeframe: {metadata}")
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
        timeout_reached = (
            current_time - trade.open_date_utc
        ).total_seconds() / 60 >= self.timeout_minutes.value

        # Log general stoploss evaluation details
        self.logger.info(
            f"[Custom Stoploss] Pair: {pair} | Timeout: {timeout_reached} | "
            f"Profit: {current_profit:.4f} | Current Rate: {current_rate:.4f} | Opened Since: {trade.open_date_utc}"
        )

        # If timeout is not reached, do not exit
        if not timeout_reached:
            self.logger.debug(
                f"[Custom Stoploss] Pair: {pair} | Timeout not reached. Holding position."
            )
            return default_stop

        # Attempt to retrieve 1h dataframe
        try:
            # 获取最新的 1h 数据
            dataframe = self.dp.get_pair_dataframe(pair=pair, timeframe="1h")
            if dataframe is None or dataframe.empty:
                raise ValueError("1h dataframe is None or empty.")

            # 计算 EMA50 如果尚未计算
            if "ema50_1h" not in dataframe.columns:
                dataframe["ema50_1h"] = ta.EMA(dataframe, timeperiod=50)

            ema_trend = dataframe["close"].iloc[-1] > dataframe["ema50_1h"].iloc[-1]
            self.logger.info(
                f"[Custom Stoploss] Pair: {pair} | EMA Trend (1h): {'Favorable' if ema_trend else 'Unfavorable'}"
            )

        except Exception as e:
            self.logger.error(
                f"[Custom Stoploss] Error retrieving or processing 1h dataframe for {pair}: {e}. Attempting to use main timeframe EMA."
            )
            # 尝试使用主时间框架的 EMA50 作为趋势判断
            try:
                dataframe = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe)
                if dataframe is None or dataframe.empty:
                    raise ValueError("Main timeframe dataframe is None or empty.")

                ema_trend_main = dataframe["close"].iloc[-1] > dataframe["ema50"].iloc[-1]
                self.logger.info(
                    f"[Custom Stoploss] Pair: {pair} | EMA Trend (main timeframe): {'Favorable' if ema_trend_main else 'Unfavorable'}"
                )
                ema_trend = ema_trend_main
            except Exception as e_main:
                self.logger.error(
                    f"[Custom Stoploss] Error retrieving or processing main timeframe dataframe for {pair}: {e_main}. Holding position."
                )
                return default_stop

        # 如果超时且趋势不利，立即退出
        if current_profit > 0 and not ema_trend:
            self.logger.warning(
                f"[Custom Stoploss] Exiting Pair: {pair} | Unfavorable Trend | Profit: {current_profit:.4f}"
            )
            return 0  # 立即退出

        # 如果超时且有最低利润，退出
        if current_profit >= self.min_profit.value:
            self.logger.warning(
                f"[Custom Stoploss] Exiting Pair: {pair} | Timeout Reached | Profit: {current_profit:.4f}"
            )
            return 0  # 立即退出

        # 如果超时但亏损，强制退出以限制亏损
        if current_profit < 0:
            self.logger.warning(
                f"[Custom Stoploss] Exiting Pair: {pair} | Timeout Reached | Loss: {current_profit:.4f}"
            )
            return 0  # 立即退出

        # 默认止损行为
        self.logger.debug(
            f"[Custom Stoploss] Pair: {pair} | Holding Position | Profit: {current_profit:.4f}"
        )
        return default_stop
