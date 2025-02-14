import os
import json
import time

import requests

if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    qald_dir = os.path.join(data_dir, 'lc-quad-2-wikidata-parafrase/test/')
    lib_dir = os.path.join(base_dir, 'lib')

    with open(os.path.join(qald_dir, 'lc_falcon_entities_test.json'), 'r') as input:
        data = json.load(input)
        for i, q in enumerate(data):
            if i % 20 ==0:
                print(f"{i} questions processed.")
            same_as = []
            if q.get("falcon_results"):
                for relation in q["falcon_results"]["relations_dbpedia"]:
                    relation, word = relation
                    url = str.encode(
                        "https://dbpedia.org/sparql?default-graph-uri=http://dbpedia.org&query=select+distinct+?a+where+{+<" + relation + ">+owl:equivalentProperty+?a++FILTER(regex(?a,'wikidata.org'))+}+&format=json")
                    response = requests.get(url)
                    print(response)
                    try:
                        result = response.json()["results"]["bindings"]
                        if len(result) > 0:
                            same_as.append([result[0]["a"]["value"], word])
                        time.sleep(2)
                    except:
                        pass
                q["falcon_results"]["equivalentProperty_DBPEDIA"] = same_as

                same_as = []
                for entitie in q["falcon_results"]["entities_dbpedia"]:
                    entitie, word = entitie
                    url = str.encode(
                        "https://dbpedia.org/sparql?default-graph-uri=http://dbpedia.org&query=select+distinct+?a+where+{+<" + entitie + ">+owl:sameAs+?a++FILTER(regex(?a,'wikidata.org'))+}+&format=json")
                    response = requests.get(url)
                    print(response)
                    try:
                        result = response.json()["results"]["bindings"]
                        if len(result) > 0:
                            same_as.append([result[0]["a"]["value"], word])
                    except:
                        pass
                    time.sleep(2)

                q["falcon_results"]["equivalentEntities_DBPEDIA"] = same_as
                data[i] = q
                print(q)
        print(data)

    with open(os.path.join(qald_dir, 'lc_falcon_entities_test_wikidata.json'), 'w') as idfile:
        json.dump(data, idfile)
