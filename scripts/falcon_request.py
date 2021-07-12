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
    qald_dir = os.path.join(data_dir, 'qald-7-wikidata')
    lib_dir = os.path.join(base_dir, 'lib')

    data = json.load(open(os.path.join(qald_dir, 'qald-7-train-en-wikidata.json')))

    questions = []
    for q in data['questions']:
        result = {}
        result["id"] = q["id"]
        result["sparql"] = q["query"]["sparql"]
        result["sparql_dummy"] = q["query"]["sparql_dummy"]
        for question in q["question"]:
            if question["language"] == "en":
                nlquestion = question["string"]
                break
        result["question"] = nlquestion
        try:
            data = str.encode('{"text":"' + nlquestion + '"}')
            print(data)
            response = requests.post('https://labs.tib.eu/falcon/falcon2/api', headers=headers, params=params,
                                     data=data)
            print(response.content)
            result["falcon_results"] = response.json()
        except:
            result["falcon_results"] = None

        time.sleep(5)
        questions.append(result)

    print(questions)

    with open(os.path.join(qald_dir, 'falcon_entities_train.json'), 'w') as idfile:
        json.dump(questions, idfile)

