import logging
import requests
import pprint

logger = logging.getLogger()
contracts = []


def get_contracts():
    response_object = requests.get("https://api-test.ascendex-sandbox.com/api/pro/v2/futures/contract", verify=False)


    for contract in response_object.json()['data']:
        contracts.append(contract['symbol'])
    return contracts



