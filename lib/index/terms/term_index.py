import json
from llama_index import Document
import pandas as pd
from lib.index.helper import cur_simple_date_time_sec

from lib.index.terms.terms import terms_from_txt
from lib import constants


doc_to_term = {}
term_to_doc = {}

def build_term_reference_index(doc: Document):
    global doc_to_term
    global term_to_doc
    terms = terms_from_txt(doc.text)
    # print(f"Document {doc.doc_id} has {len(terms)} terms ..." + doc.text[:100] + " ...")
    for term in terms:
        if term not in term_to_doc:
            term_to_doc[term] = {}
        term_to_doc[term][doc.doc_id] = term_to_doc[term].get(doc.doc_id, 0) + 1

        if doc.doc_id not in doc_to_term:
            doc_to_term[doc.doc_id] = {}
        doc_to_term[doc.doc_id][term] = doc_to_term[doc.doc_id].get(term, 0) + 1

def write_term_references_to_file():
    time = cur_simple_date_time_sec()
    with open(f"{constants.term_data_dir}/term_to_doc_{time}.json", "w") as f:
        json.dump(term_to_doc, f, indent=2)
    with open(f"{constants.term_data_dir}/doc_to_term_{time}.json", "w") as f:
        json.dump(doc_to_term, f, indent=2)

def count_terms_per_document():
    term_count = []
    for term, doc_counts in get_term_to_doc_items():
        num_docs = len(doc_counts)
        num_all = sum(doc_counts.values())
        term_count.append((term, num_docs, num_all))
    tpd = pd.DataFrame(term_count, columns=["term", "num_docs", "num_all"])
    tpd = tpd.sort_values(by="num_all", ascending=False)
    tpd.to_csv(f"{constants.term_data_dir}/term_count_{cur_simple_date_time_sec()}.csv", index=False)
    print(tpd)

def get_term_to_doc_items():
    return term_to_doc.items()