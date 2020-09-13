from SPARQLWrapper import SPARQLWrapper, JSON
import json
import os
import time
import tqdm


def loadItemToKGMap(filename):
    # open DBpedia connector dictionary
    with open(filename, 'r') as fin:
        item_to_kg_dict = {}
        for line in fin:
            # split each line and remove titles with fewer than 3 values
            line_split = line.strip().split('\t')
            if len(line_split) < 3:
                continue

            # add the remaining titles to the item to kg dictionary
            item_id = line_split[0]
            db_uri = line_split[2]
            item_to_kg_dict[item_id] = db_uri

    return item_to_kg_dict


def getHeadQuery(entity):
    # get DBpedia head query
    return "SELECT * WHERE { <%s> ?p ?o }" % entity


def getTailQuery(entity):
    # get DBpedia tail query
    return "SELECT * WHERE { ?s ?p <%s> }" % entity


def cleanHeadResults(results):
    # initialize local list and sets
    results_cl = []
    predicate_set = set()
    entity_set = set()

    # for each head query, clean its results
    for result in results["results"]["bindings"]:
        # skip those non english title head queries
        if result['o']['type'] == 'literal' and 'xml:lang' in result['o'] and \
                result['o']['xml:lang'] != 'en':
            continue

        # clean results for english title head queries
        if result['o']['type'] == 'uri':
            entity_set.add(result['o']['value'])
        predicate_set.add(result['p']['value'])
        results_cl.append(result)
    return results_cl, predicate_set, entity_set


def cleanTailResults(results):
    # initialize local list and sets
    results_cl = []
    predicate_set = set()
    entity_set = set()

    # for each tail query, clean its results
    for result in results["results"]["bindings"]:
        entity_set.add(result['s']['value'])
        predicate_set.add(result['p']['value'])
        results_cl.append(result)
    return results_cl, predicate_set, entity_set


def downloadDBPedia(sparql, fout, entities, asTail=True):
    print("Querying {} Titles".format(len(entities)))
    pbar = tqdm.tqdm(total=len(entities))
    sec_to_wait = 60
    
    for entity in entities:
        pbar.update(1)
        while True:
            try:
                sparql.setQuery(getHeadQuery(entity))
                head_results = sparql.query().convert()
                break
            except:
                time.sleep(sec_to_wait)

        # clean head query results 
        head_results_cl, predicate_set, entity_set = cleanHeadResults(head_results)
        head_json_str = json.dumps(head_results_cl)

        if asTail:
            while True:
                try:
                    sparql.setQuery(getTailQuery(entity))
                    tail_results = sparql.query().convert()
                    break
                except:
                    # print("http failure! wait %d seconds to retry..." % sec_to_wait)
                    time.sleep(sec_to_wait)
            tail_results_cl, tail_predicate_set, tail_entity_set = cleanTailResults(tail_results)
            tail_json_str = json.dumps(tail_results_cl)
            predicate_set |= tail_predicate_set
            entity_set |= tail_entity_set

        fout.write(entity + '\t' + head_json_str + '\t' + tail_json_str + '\n')
        # print("finish! {} entities and {} predicates!".format(len(entity_set), len(predicate_set)))
        time.sleep(1)

    print("finished querying all titles!")
    return entity_set, predicate_set


if __name__ == "__main__":
    # set sql querying object
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)

    # set file paths
    item2kg_file = "source/ml1m/MappingMovielens2DBpedia-1.2.tsv"
    kg_path = "source/ml1m/kg/"

    # initialize predicate and item sets
    all_predicate_set = set()
    all_entity_set = set()

    # load connector dictionary and update entity set
    item2kg_dict = loadItemToKGMap(item2kg_file)
    item_entities = set(item2kg_dict.values())
    all_entity_set.update(item_entities)

    # load kg data file
    kg_file = os.path.join(kg_path, "kg.dat")
    with open(kg_file, 'a') as fout:
        # download DBpedia results and remove entities found
        entity_set, predicate_set = downloadDBPedia(sparql, fout, item_entities, asTail=True)

        # remove all items not found
        all_predicate_set |= predicate_set
        all_entity_set |= entity_set

    # load predicate file and save query results
    predicate_file = os.path.join(kg_path, "predicate_vocab.dat")
    with open(predicate_file, 'w') as fout:
        for pred in all_predicate_set:
            fout.write(pred + '\n')

    # load entity file and save results
    entity_file = os.path.join(kg_path, "entity_vocab.dat")
    with open(entity_file, 'w') as fout:
        for ent in all_entity_set:
            fout.write(ent + '\n')
