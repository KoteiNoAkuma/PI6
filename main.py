from pybit.unified_trading import HTTP

if __name__ == '__main__':
    session = HTTP(testnet=True)
    print(session.get_public_trade_history(
        category="spot",
        symbol="BTCUSDT",
        limit=1,
    ))