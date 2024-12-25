from typing import Dict
from pandas import DataFrame
import logging
import talib.abstract as ta
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from freqtrade.persistence import Trade
from technical import qtpylib
from decimal import Decimal


class ComplexStrategy(IStrategy):
    """
    Enhanced strategy:
    - Exit immediately if timeout is reached and the trend is unfavorable.
    - ROI standards still apply.
    - Stay in trades with favorable trends even after timeout.
    - Considers both slippage and fees in the calculations.
    """

    INTERFACE_VERSION = 3

    # No shorting
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
        low=0.005, high=0.03, decimals=4, default=0.0075, space="custom", optimize=True
    )

    # Parameterized slippage
    slippage = DecimalParameter(
        0.0, 0.005, decimals=4, default=0.001, space="risk", optimize=True
    )  # 0.1%

    # Fixed fees for normal users on Binance
    maker_fee = Decimal("0.001")  # 0.1%
    taker_fee = Decimal("0.001")  # 0.1%

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

    def simulate_slippage(self, price: float, is_buy: bool, slippage: float) -> float:
        """
        Simulate slippage by adjusting the price.
        """
        if is_buy:
            return price * (1 + slippage)
        else:
            return price * (1 - slippage)

    def calculate_fees(self, amount: float, fee: Decimal) -> float:
        """
        Calculate fees based on the amount and fee rate.
        """
        return float(Decimal(amount) * fee)

    def custom_stoploss(
        self, pair: str, trade: Trade, current_time, current_rate, current_profit, **kwargs
    ) -> float:
        """
        Custom exit logic:
        - Exit immediately if timeout is reached and the trend is unfavorable.
        - Stay in position if the trend is favorable, even after timeout.
        - Considers both slippage and fees in the calculations.
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

        # 获取当前交易的买入和卖出手续费
        trade_fee_open = self.calculate_fees(trade.open_rate * trade.amount, self.maker_fee)
        trade_fee_close = self.calculate_fees(current_rate * trade.amount, self.taker_fee)

        # 获取滑点值
        current_slippage = self.slippage.value

        # 调整当前利润以考虑滑点和手续费
        # 假设是买入滑点和卖出滑点都需要考虑
        adjusted_current_rate_buy = self.simulate_slippage(
            trade.open_rate, is_buy=True, slippage=current_slippage
        )
        adjusted_current_rate_sell = self.simulate_slippage(
            current_rate, is_buy=False, slippage=current_slippage
        )

        # 计算调整后的利润
        # (卖出价格 - 买入价格) * 数量 - 手续费
        adjusted_profit = (
            adjusted_current_rate_sell - adjusted_current_rate_buy
        ) * trade.amount - (trade_fee_open + trade_fee_close)

        self.logger.info(
            f"[Custom Stoploss] Pair: {pair} | Adjusted Profit: {adjusted_profit:.4f} | "
            f"Original Profit: {current_profit:.4f}"
        )

        # 如果超时且利润减去滑点和手续费后仍然盈利，但趋势不利，退出
        if adjusted_profit > 0 and not ema_trend:
            self.logger.warning(
                f"[Custom Stoploss] Exiting Pair: {pair} | Unfavorable Trend | Adjusted Profit: {adjusted_profit:.4f}"
            )
            return 0  # 立即退出

        # 如果超时且调整后的利润达到最低利润，退出
        if adjusted_profit >= self.min_profit.value:
            self.logger.warning(
                f"[Custom Stoploss] Exiting Pair: {pair} | Timeout Reached | Adjusted Profit: {adjusted_profit:.4f}"
            )
            return 0  # 立即退出

        # 如果超时但亏损，强制退出以限制亏损
        if adjusted_profit < 0:
            self.logger.warning(
                f"[Custom Stoploss] Exiting Pair: {pair} | Timeout Reached | Adjusted Loss: {adjusted_profit:.4f}"
            )
            return 0  # 立即退出

        # 默认止损行为
        self.logger.debug(
            f"[Custom Stoploss] Pair: {pair} | Holding Position | Adjusted Profit: {adjusted_profit:.4f}"
        )
        return default_stop

    def populate_entry_order(self, pair: str, order: dict, trade: Trade, **kwargs):
        """
        Override to apply slippage and fees to entry orders.
        """
        if order["side"] == "buy":
            # Apply slippage to buy price
            order["price"] = self.simulate_slippage(
                order["price"], is_buy=True, slippage=self.slippage.value
            )
            # Calculate and log maker fee
            fee = self.calculate_fees(order["price"] * order["amount"], self.maker_fee)
            self.logger.debug(f"Buy Order Adjusted Price: {order['price']:.4f}, Fee: {fee:.4f}")
        elif order["side"] == "sell":
            # Apply slippage to sell price
            order["price"] = self.simulate_slippage(
                order["price"], is_buy=False, slippage=self.slippage.value
            )
            # Calculate and log taker fee
            fee = self.calculate_fees(order["price"] * order["amount"], self.taker_fee)
            self.logger.debug(f"Sell Order Adjusted Price: {order['price']:.4f}, Fee: {fee:.4f}")

        return order

    def notify_trade(self, trade: Trade, order: dict, **kwargs) -> None:
        """
        Override to log trade details including fees and slippage.
        """
        if trade.is_open:
            self.logger.info(
                f"Opened trade #{trade.trade_id} | Pair: {trade.pair} | Amount: {trade.amount} | Open Rate: {trade.open_rate:.4f}"
            )
        else:
            self.logger.info(
                f"Closed trade #{trade.trade_id} | Pair: {trade.pair} | Amount: {trade.amount} | "
                f"Open Rate: {trade.open_rate:.4f} | Close Rate: {trade.close_rate:.4f} | "
                f"Profit: {trade.pnl:.4f} | Fees: {trade.fees:.4f}"
            )
