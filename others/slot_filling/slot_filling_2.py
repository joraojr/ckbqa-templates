import os


def get_wikidata_entites(question):
    import spacy
    from spacy.language import Language

    from spacyEntityLinker import EntityLinker

    # initialize language model
    nlp = spacy.load("en_core_web_sm")

    def create_etitylinker(nlp, name):
        return EntityLinker()

    Language.factory("entityLinker", func=create_etitylinker)

    # add pipeline
    nlp.add_pipe("entityLinker", last=True)

    doc = nlp(question)

    # returns all entities in the whole document
    all_linked_entities = doc._.linkedEntities

    results = []
    for x in all_linked_entities:
        results.append(["https://www.wikidata.org/wiki/Q" + str(x.get_id()), str(x.get_span()), ])

    return results


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    qald_dir = os.path.join(data_dir, 'qald-7-wikidata')

    json_object = []
    import json
    import pandas as pd

    templates = pd.read_csv(os.path.join(qald_dir, 'templates.csv'))
    with open(os.path.join(qald_dir, 'falcon_entities_train_wikidata.json'), 'r') as inputfile:
        prediction = json.load(inputfile)
        for item in prediction:
            item_template = []

            item['wikidata_entities'] = get_wikidata_entites(item["question"])
            json_object.append(item)

        print(json.dumps(json_object))
