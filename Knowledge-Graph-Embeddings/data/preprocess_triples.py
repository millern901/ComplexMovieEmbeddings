import json
import os
import random
import math
import logging


class Triple(object):
    def __init__(self, head, relation, tail):
        self.h = head
        self.r = relation
        self.t = tail


def preprocess(source_path, preprocess_path, entity_file=None, relation_file=None, map_file=None, train_ratio=0.7,
               test_ratio=0.2, shuffle_data_split=True, filter_unseen_samples=True, low_frequency=10, out_logger=None):
    # create train/test/validate file paths
    train_file = os.path.join(preprocess_path, "train.txt")
    test_file = os.path.join(preprocess_path, "test.txt")
    valid_file = os.path.join(preprocess_path, "valid.txt")
    # create mapping files
    e_map_file = os.path.join(preprocess_path, "entities.dict")
    r_map_file = os.path.join(preprocess_path, "relations.dict")

    # log beginning of the preprocess of the kg data
    out_logger.info("shuffle and split {} for {:.1f} training, {:.1f} validation and {:.1f} testing!"
                    .format(source_path, train_ratio, 1 - train_ratio - test_ratio, test_ratio))

    # set predefined entity vocab for filtering
    ent_keep_vocab = set()
    with open(entity_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            line_split = line.strip().split('\t')
            if len(line_split) < 3:
                continue
            ent_keep_vocab.add(line_split[2])
    fin.close()
    # set predefined relation vocab for filtering
    relation_vocab = set()
    with open(relation_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            relation_vocab.add(line.strip())
    fin.close()

    # load the kg data
    triple_list, entity_dict = loadRawData(file_name=source_path, relation_vocab=relation_vocab, out_logger=out_logger)

    # filter low frequent entities
    filtered_triple_list, e_set, r_set = cutLowFrequentData(triple_list=triple_list, entity_frequency_dict=entity_dict,
                                                            ent_vocab_to_keep=ent_keep_vocab, low_frequency=low_frequency,
                                                            out_logger=out_logger)

    # split the kg data into test and train
    train, valid, test, e_map, r_map = splitKGData(triple_list=filtered_triple_list, train_ratio=train_ratio, test_ratio=test_ratio,
                                                   map_file=map_file, filter_unseen_samples=filter_unseen_samples, out_logger=out_logger)

    out_logger.info("{} entities and {} relations, where {} train, {} valid, and {} test!"
                    .format(len(e_map), len(r_map), len(train), len(valid), len(test)))
    out_logger.info("Saving dictionaries and model files!")

    # save ent_dic, rel_dic
    with open(e_map_file, 'w', encoding='utf-8') as fout:
        for uri in e_map:
            fout.write('{}\t{}\n'.format(e_map[uri], uri.strip()))
    with open(r_map_file, 'w', encoding='utf-8') as fout:
        for uri in r_map:
            fout.write('{}\t{}\n'.format(r_map[uri], uri))

    # save train/test/validation files
    with open(train_file, 'w', encoding='utf-8') as fout:
        for triple in train:
            fout.write('{}\t{}\t{}\n'.format(triple.h, triple.r, triple.t))
    with open(test_file, 'w', encoding='utf-8') as fout:
        for triple in test:
            fout.write('{}\t{}\t{}\n'.format(triple.h, triple.r, triple.t))
    with open(valid_file, 'w', encoding='utf-8') as fout:
        for triple in valid:
            fout.write('{}\t{}\t{}\n'.format(triple.h, triple.r, triple.t))


def loadRawData(file_name, relation_vocab=None, out_logger=None):
    # log the number of relations inside of the vocab
    out_logger.info("Predefined vocab: use {} relations in vocab!".format(len(relation_vocab)))

    triple_list = list()
    entity_dict = dict()
    with open(file_name, 'r', encoding='utf-8') as fin:
        for line in fin:
            # split each line and check for valid length
            line_split = line.strip().split('\t')
            if len(line_split) < 3:
                continue

            # grab entity and update frequency
            e = line_split[0]
            entity_dict[e] = entity_dict[e] + 1 if e in entity_dict.keys() else 1

            # grab the head and tail queries from the line
            head_json_list = json.loads(line_split[1])
            tail_json_list = json.loads(line_split[2])

            for head_json in head_json_list:
                rt = parseRT(head_json, relation_set=relation_vocab)
                if rt is None:
                    continue
                r, t = rt
                entity_dict[t] = entity_dict[t] + 1 if t in entity_dict.keys() else 1
                triple_list.append((e, r, t))

            for tail_json in tail_json_list:
                hr = parseHR(tail_json, relation_set=relation_vocab)
                if hr is None:
                    continue
                h, r = hr
                entity_dict[h] = entity_dict[h] + 1 if h in entity_dict.keys() else 1

                triple_list.append((e, r, t))

    # log the end of loading the data
    out_logger.info("Totally {} facts of {} entities from {}!".format(len(triple_list), len(entity_dict), file_name))

    return triple_list, entity_dict


def parseRT(json_dict, relation_set=None):
    r = json_dict['p']['value']
    t_type = json_dict['o']['type']
    t = json_dict['o']['value']
    if t_type != 'uri' or (relation_set is not None and r not in relation_set):
        return None
    return r, t


def parseHR(json_dict, relation_set=None):
    r = json_dict['p']['value']
    h = json_dict['s']['value']
    if relation_set is not None and r not in relation_set:
        return None
    return h, r


def cutLowFrequentData(triple_list, entity_frequency_dict, ent_vocab_to_keep=None, low_frequency=10, out_logger=None):
    # initialize temporary entity and relation sets
    tmp_entity_set = set()
    tmp_relation_set = set()

    filtered_triple_list = []
    for triple in triple_list:
        if ((entity_frequency_dict[triple[0]] >= low_frequency and entity_frequency_dict[triple[2]] >= low_frequency)
                or (triple[0] in ent_vocab_to_keep and triple[2] in ent_vocab_to_keep)
                or (triple[0] in ent_vocab_to_keep and entity_frequency_dict[triple[2]] >= low_frequency)
                or (entity_frequency_dict[triple[0]] >= low_frequency and triple[2] in ent_vocab_to_keep)):
            filtered_triple_list.append(triple)
            tmp_entity_set.add(triple[0])
            tmp_entity_set.add(triple[2])
            tmp_relation_set.add(triple[1])

    # log finish of cutting low frequency entities
    out_logger.info("Cut infrequent entities (<={}), remaining {} facts of {} entities and {} relations!"
                    .format(low_frequency, len(filtered_triple_list), len(tmp_entity_set), len(tmp_relation_set)))

    return filtered_triple_list, tmp_entity_set, tmp_relation_set


def splitKGData(triple_list, train_ratio=0.7, test_ratio=0.1, map_file=None, filter_unseen_samples=True, out_logger=None):
    # valid ratio could be 1-train_ratio-test_ratio, and maybe zero
    assert 0 < train_ratio < 1, "train ratio out of range!"
    assert 0 < test_ratio < 1, "test ratio out of range!"
    valid_ratio = 1 - train_ratio - test_ratio
    assert 0 <= valid_ratio < 1, "valid ratio out of range!"

    train_ent_set = set()
    train_rel_set = set()
    random.shuffle(triple_list)

    n_total = len(triple_list)
    n_train = math.ceil(n_total * train_ratio)
    n_valid = math.ceil(n_total * valid_ratio) if valid_ratio > 0 else 0

    # in case of zero test item
    if n_train >= n_total:
        n_train = n_total - 1
        n_valid = 0
    elif n_train + n_valid >= n_total:
        n_valid = n_total - 1 - n_train

    tmp_train_list = [i for i in triple_list[0:n_train]]
    tmp_valid_list = [i for i in triple_list[n_train:n_train + n_valid]]
    tmp_test_list = [i for i in triple_list[n_train + n_valid:]]

    for triple in tmp_train_list:
        train_ent_set.add(triple[0])
        train_ent_set.add(triple[2])
        train_rel_set.add(triple[1])

    out_logger.info("Splitting of data into train/test/validation is done!")

    # initialize entity map
    entity_map = {}
    with open(map_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            items = line.strip().split('\t')
            entity_map[items[1].strip()] = items[0]
    fin.close()

    for index, ent in enumerate(train_ent_set):
        if ent.strip() in entity_map.keys():
            continue
        else:
            entity_map[ent] = len(entity_map)

    r_map = {}
    for index, rel in enumerate(train_rel_set):
        r_map[rel] = index

    train_list = [Triple(triple[0], triple[2], triple[1]) for triple in tmp_train_list]

    out_logger.info("Filtering unseen entities and relations.")

    if filter_unseen_samples:
        valid_list = [Triple(triple[0], triple[2], triple[1]) for triple in tmp_valid_list if
                      triple[0] in train_ent_set and triple[2] in train_ent_set and triple[1] in train_rel_set]
        test_list = [Triple(triple[0], triple[2], triple[1]) for triple in tmp_test_list if
                     triple[0] in train_ent_set and triple[2] in train_ent_set and triple[1] in train_rel_set]
    else:
        valid_list = [Triple(triple[0], triple[2], triple[1]) for triple in tmp_valid_list]
        test_list = [Triple(triple[0], triple[2], triple[1]) for triple in tmp_test_list]

    return train_list, valid_list, test_list, entity_map, r_map


if __name__ == "__main__":
    # set main directory and dataset values
    in_head_path = "data/source"
    out_head_path = "data/preprocess"
    dataset = 'ml1m'
    data_type = 'kg'

    # set data in path
    in_path = os.path.join(in_head_path, dataset)
    data_in_path = os.path.join(in_path, data_type)

    # set data out paths
    out_path = os.path.join(out_head_path, dataset)
    data_out_path = os.path.join(out_path, data_type)

    # set data files
    triple_file_path = os.path.join(data_in_path, "kg.dat")
    relation_file_path = os.path.join(data_in_path, "predicate_vocab.dat")
    entity_file_path = os.path.join(in_path, "MappingMovielens2DBpedia-1.2.tsv")
    item_dict_path = os.path.join(out_path, "rating", "item.dict")

    # initialize logger
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    log_file = os.path.join(out_path, "kg_data_preprocess.log")
    # set logger format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # set logger file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # set logger stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    preprocess(source_path=triple_file_path, preprocess_path=data_out_path, entity_file=entity_file_path,
               relation_file=relation_file_path, map_file=item_dict_path, train_ratio=0.7, test_ratio=0.2,
               shuffle_data_split=True, filter_unseen_samples=True, low_frequency=10, out_logger=logger)
