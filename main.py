
import logging

from connectors.binance_futures  import BinanceFuturesClient
from connectors.bitmex import BitmexClient

from interface.root_component import Root



logger = logging.getLogger()

logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s :: %(message)s')
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler('info.log')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)


if __name__ == '__main__':

    binance = BinanceFuturesClient("822defebe18516b842c7e06ba41a51b0ac51b21f6e5c031426283ad12d7825fe",
                                   "340ce454ae055549b974df919430a5b22e41328a1a706ba454f3457ee4fe0459",True)

    bitmex = BitmexClient("ABjJHe7MqLyfXCNFcQ3KNxdl",
                          "ur27aM3RGD0hDfiE5LzDyvjgTzi9_2mAt3REqKDFdIxXuI51", True)

  #  print(binance.place_order(binance.contracts['BTCUSDT'],"Buy",50,"Limit",price=20000,tif=None))

   # print(bitmex.place_order(bitmex.contracts['XBTUSD'], "Limit", 60, "Buy",price=20000.4142412,tif="GoodTillCancel"))

   #testing
    #print(bitmex.contracts['XBTUSD'].base_asset, bitmex.contracts['XBTUSD'].price_decimals)
    #print(bitmex.balances['XBt'].wallet_balance)


   # print(bitmex.cancel_order("38839370-2578-4b22-9ebc-c346d07c32d2").status)
   # print(bitmex.get_order_status("38839370-2578-4b22-9ebc-c346d07c32d2", bitmex.contracts['XBTUSD']).status)

   # bitmex.get_historical_candles(bitmex.contracts['XBTUSD'],"1h")

    root = Root(binance,bitmex)
    root.mainloop()


