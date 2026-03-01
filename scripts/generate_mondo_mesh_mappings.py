import re
import obonet
from gilda.process import normalize
from gilda.generate_terms import generate_mesh_terms
from indra.databases import mesh_client
import pandas as pd
from biomappings.resources import PREDICTIONS_SSSOM_PATH, POSITIVES_SSSOM_PATH

MONDO_URL = ('https://raw.githubusercontent.com/monarch-initiative/mondo/'
             'refs/heads/master/src/ontology/mondo-edit.obo')

if __name__ == '__main__':
    # Load the latest MONDO ontology
    g = obonet.read_obo(MONDO_URL)

    # Get only diseases from MeSH
    mesh_disease_ids = {
        mesh_id for mesh_id in mesh_client.mesh_id_to_name
        if mesh_client.has_tree_prefix(mesh_id, 'C')
    }

    # Get the normalized MeSH names and synonyms from Gilda
    gilda_mesh_terms = [
        term for term in generate_mesh_terms(ignore_mappings=True)
        if term.id in mesh_disease_ids
    ]

    # Initialize multiple mappings data structures
    # Mappings in both directions
    mondo_to_mesh = defaultdict(set)
    mesh_to_mondo = defaultdict(set)

    # Keep track of things that haven't already been mapped
    # in either direction
    mondo_no_mesh_mapping = set([node for node in g
                                 if node.startswith('MONDO')])
    mesh_no_mondo_mapping = {t for t in mesh_disease_ids}

    # Keep track of MONDO terms by Gilda-normalized lexicalization
    mondo_terms_by_norm = defaultdict(set)

    # Populate the data structures from MONDO to know what mappings
    # already exist from the primary source, and simulataneously
    # get all the lexicalications from MONDO
    for node in g:
        if not node.startswith('MONDO'):
            continue
        node_data = g.nodes[node]
        if not node_data:
            continue
        xrefs = node_data.get('xref', [])
        for xref in xrefs:
            if xref.startswith('MESH'):
                has_mesh_mapping = True
                mondo_to_mesh[node].add(xref)
                mesh_to_mondo[xref].add(node)
                mesh_no_mondo_mapping -= {xref}
                mondo_no_mesh_mapping -= {node}
        mondo_terms_by_norm[normalize(node_data['name'])].add(node)
        for synonym in node_data.get('synonym', []):
            match = re.match(r'^\"(.+)\" (EXACT|RELATED|NARROW|BROAD|\[\])',
                             synonym)
            syn, status = match.groups()
            if status == 'EXACT':
                mondo_terms_by_norm[normalize(syn)].add(node)

    # Keep track of MESH terms by Gilda-normalized lexicalization
    mesh_terms_by_norm = defaultdict(set)
    for term in gilda_mesh_terms:
        mesh_terms_by_norm[term.norm_text].add(term.id)

    # Find which Gilda-normalized lexicalizations overlap, i.e.,
    # the broadest definition of possible mappings
    overlap_norms = set(mondo_terms_by_norm) & set(mesh_terms_by_norm)

    # Filter all overlaps to unambiguous mappings in both directions
    unambig_overlap_norms = {
        n for n in overlap_norms
        if (len(mondo_terms_by_norm[n]) == 1 and
            len(mesh_terms_by_norm[n]) == 1)
    }

    # Filter unambiguous mappings to ones where we make sure that
    # nothing to-from the given terms has already been mapped, i.e.,
    # the mapping would be novel in that the given MONDO entry as no
    # mapping to MESH and there is no mapping from any other MONDO
    # term to the given MESH term
    novel_unambig_overlap_norms = {
        n: (list(mondo_terms_by_norm[n])[0], list(mesh_terms_by_norm[n])[0])
            for n in unambig_overlap_norms if
           (list(mondo_terms_by_norm[n])[0] in mondo_no_mesh_mapping
            and list(mesh_terms_by_norm[n])[0] in mesh_no_mondo_mapping)
    }

    # Now filter out anything that we've already predicted or curated
    # as part of Biomappings
    df_pred = pd.read_csv(PREDICTIONS_SSSOM_PATH, comment='#', sep='\t')
    df_pos = pd.read_csv(POSITIVES_SSSOM_PATH, comment='#', sep='\t')

    def get_mappings_for_pair(df, ns1, ns2, predicate='skos:exactMatch'):
        mappings = []
        df12 = df[df['subject_id'].str.startswith(ns1) &
                  df['object_id'].str.startswith(ns2) &
                  (df['predicate_id'] == predicate)]
        for _, row in df12.iterrows():
            mappings.append((row['subject_id'], row['object_id']))
        df21 = df[df['subject_id'].str.startswith(ns2) &
                  df['object_id'].str.startswith(ns1) &
                  (df['predicate_id'] == predicate)]
        for _, row in df21.iterrows():
            mappings.append((row['object_id'], row['subject_id']))
        return mappings

    pred_mappings = get_mappings_for_pair(df_pred, 'mondo', 'mesh')
    pos_mappings = get_mappings_for_pair(df_pos, 'mondo', 'mesh')

    pred_mappings = [(mondo_id.upper(), mesh_id.split(':')[1]) for
                     mondo_id, mesh_id in pred_mappings]
    pos_mappings = [(mondo_id.upper(), mesh_id.split(':')[1]) for
                     mondo_id, mesh_id in pos_mappings]

    novel_unambig_overlap_norms_no_biomappings = {
        k: v for k, v in novel_unambig_overlap_norms.items()
        if (v not in pred_mappings) and (v not in pos_mappings)
    }

    # Finally, specifically highlight terms that are relevant for HLBS
    # using the MeSH structure
    hlbs_mesh_codes = {
        'C14', # Cardiovascular Diseases
        'C08', # Respiratory Tract Diseases (includes lung, etc)
        'C15.378', # Hematologic Diseases
        'C10.886', # Sleep Disorders
    }
    hlbs_novel_unambig_norms = {}
    for mesh_tree_code in hlbs_mesh_codes:
        hlbs_novel_unambig_norms.update(
            {k: v for k, v in novel_unambig_overlap_norms.items()
             if mesh_client.has_tree_prefix(v[1], mesh_tree_code)})
