if __name__ == "__main__":
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

    doc = nlp("Which presidents were born in 1945?")

    # returns all entities in the whole document
    all_linked_entities = doc._.linkedEntities
    # iterates over sentences and prints linked entities
    for sent in doc.sents:
        sent._.linkedEntities.pretty_print()
        print("\n")
