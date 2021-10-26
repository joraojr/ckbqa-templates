import time

import requests
import os

if __name__ == "__main__":
    import requests
    import json

    headers = {
        'Content-Type': 'application/json',
    }

    params = (
        ('mode', 'long'),
        ('db', '1'),
    )

    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    lc_quadir = os.path.join(data_dir, "lc-quad-2-wikidata-parafrase/test/")
    # qald_dir = os.path.join(data_dir, 'qald-7-wikidata')
    # lib_dir = os.path.join(base_dir, 'lib')
    #
    # data = json.load(open(os.path.join(qald_dir, 'qald-7-train-en-wikidata.json')))

    questions = []
    with open(os.path.join(lc_quadir, 'input.txt'), 'r') as inputfile, \
            open(os.path.join(lc_quadir, 'id.txt'), 'r') as idfile, \
            open(os.path.join(lc_quadir, 'output.txt'), 'r') as outputfile, \
            open(os.path.join(lc_quadir, 'output_group.txt'), 'r') as output_groupfile:
        for i, lines in enumerate(
                zip(inputfile.readlines(), idfile.readlines(), outputfile.readlines(), output_groupfile.readlines())):
            result = {}
            nlquestion, id, output, output_group = lines
            nlquestion = nlquestion.strip()
            id = id.strip()
            output = output.strip()
            output_group = output_group.strip()
            result["id"] = id
            result["question"] = nlquestion
            result["sparql_dummy_id"] = output
            result["sparql_group"] = output_group
            if i % 20 == 0:
                print(f"{i} questions processed")
            try:
                data = str.encode('{"text":"' + nlquestion + '"}')
                print(data)
                response = requests.post('https://labs.tib.eu/falcon/falcon2/api', headers=headers, params=params,
                                         data=data)
                print(response.content)
                result["falcon_results"] = response.json()
            except:
                result["falcon_results"] = None

            time.sleep(1)
            questions.append(result)

    # for q in data['questions']:
    #     result = {}
    #     result["id"] = q["id"]
    #     result["sparql"] = q["query"]["sparql"]
    #     result["sparql_dummy"] = q["query"]["sparql_dummy"]
    #     for question in q["question"]:
    #         if question["language"] == "en":
    #             nlquestion = question["string"]
    #             break
    #     result["question"] = nlquestion
    #     try:
    #         data = str.encode('{"text":"' + nlquestion + '"}')
    #         print(data)
    #         response = requests.post('https://labs.tib.eu/falcon/falcon2/api', headers=headers, params=params,
    #                                  data=data)
    #         print(response.content)
    #         result["falcon_results"] = response.json()
    #     except:
    #         result["falcon_results"] = None
    #
    #     time.sleep(5)
    #     questions.append(result)
    #
    # print(questions)

    with open(os.path.join(lc_quadir, 'lc_falcon_entities_test.json'), 'w') as idfile:
        json.dump(questions, idfile)
