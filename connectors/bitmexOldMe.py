import logging
import requests
import pprint

logger = logging.getLogger()
contracts = []


def get_bitmexcontracts():
    response_object = requests.get("https://www.bitmex.com/api/v1/instrument/active")

    for contract in response_object.json():
        contracts.append(contract['symbol'])
    return contracts



