import time

import pandas as pd
import requests

test_json = pd.read_json("datasets/df_test_novo.json")
dummy_json = pd.read_json("../data/lc-quad-2-wikidata-parafrase/DummyTemplatesWikidata.json")


def get_sparql_wikidata(x):
    x = dummy_json[dummy_json.uid == x]
    return x.sparql_wikidata.item()


def get_sparql_answer(x):
    x = x.strip()
    url = str.encode(
        f"https://query.wikidata.org/sparql?query={x}&format=json")
    response = requests.get(url)
    print(x)
    print(response)
    try:
        print(response.json())
        answer = response.json()  # ["results"]["bindings"]
        time.sleep(2)
    except:
        answer = None
    return answer


# test_json["sparql_wikidata"] = None

test_json["sparql_wikidata"] = test_json.uid.apply(get_sparql_wikidata)

test_json["answer"] = test_json.sparql_wikidata.apply(get_sparql_answer)

test_json.to_json("./df_test_with_sparql.json", orient="records", default_handler=str)
