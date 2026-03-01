import os
import re
from collections import defaultdict

import obonet
import pandas as pd
from indra.databases import mesh_client
from biomappings.resources import PREDICTIONS_SSSOM_PATH, POSITIVES_SSSOM_PATH

MONDO_URL = ('https://raw.githubusercontent.com/monarch-initiative/mondo/'
             'refs/heads/master/src/ontology/mondo-edit.obo')

LOGMAP_INPUT = os.path.join(
    os.path.dirname(__file__),
    'all_mondo_to_mesh_mappings_logmap_26_02_25.tsv')


if __name__ == '__main__':
    # Load the latest MONDO ontology
    g = obonet.read_obo(MONDO_URL)

    # Load LogMap predictions
    df_logmap = pd.read_csv(LOGMAP_INPUT, sep='\t')

    # Build sets of existing MONDO-MeSH mappings from MONDO xrefs
    # to determine which LogMap predictions are truly novel
    mondo_with_mesh = set()
    mesh_with_mondo = set()
    for node in g:
        if not node.startswith('MONDO'):
            continue
        node_data = g.nodes[node]
        if not node_data:
            continue
        for xref in node_data.get('xref', []):
            if xref.startswith('MESH'):
                # xrefs are like "MESH:D003866", extract bare ID
                mesh_id = xref.split(':')[1]
                mondo_with_mesh.add(node)
                mesh_with_mondo.add(mesh_id)

    # Extract MONDO and MeSH IDs from LogMap output
    # source identifier is mondo CURIE (e.g., "mondo:0002050")
    # target identifier is mesh CURIE (e.g., "mesh:D003866")
    logmap_mappings = []
    for _, row in df_logmap.iterrows():
        mondo_curie = row['source identifier']  # e.g., "mondo:0002050"
        mesh_curie = row['target identifier']    # e.g., "mesh:D003866"
        # Convert to the uppercase format used in the OBO graph
        mondo_obo = mondo_curie.upper().replace(':', ':')  # "MONDO:0002050"
        mesh_id = mesh_curie.split(':')[1]                 # "D003866"
        logmap_mappings.append({
            'mondo_curie': mondo_curie,
            'mondo_obo': mondo_obo,
            'mesh_curie': mesh_curie,
            'mesh_id': mesh_id,
            'confidence': row['confidence'],
            'mesh_label': row['target name'],
            'mondo_label': row['source name'],
        })

    # Filter to unambiguous mappings: each MONDO ID and each MeSH ID
    # should appear at most once across all LogMap predictions
    mondo_counts = defaultdict(int)
    mesh_counts = defaultdict(int)
    for m in logmap_mappings:
        mondo_counts[m['mondo_obo']] += 1
        mesh_counts[m['mesh_id']] += 1

    unambig_mappings = [
        m for m in logmap_mappings
        if mondo_counts[m['mondo_obo']] == 1
        and mesh_counts[m['mesh_id']] == 1
    ]
    print(f"Total LogMap mappings: {len(logmap_mappings)}")
    print(f"Unambiguous (1:1) mappings: {len(unambig_mappings)}")

    # Filter to novel mappings where neither the MONDO term already
    # has a MeSH xref nor the MeSH term already has a MONDO xref
    novel_mappings = [
        m for m in unambig_mappings
        if m['mondo_obo'] not in mondo_with_mesh
        and m['mesh_id'] not in mesh_with_mondo
    ]
    print(f"Novel mappings (no existing xref): {len(novel_mappings)}")

    # Filter out anything already predicted or curated in Biomappings
    df_pred = pd.read_csv(PREDICTIONS_SSSOM_PATH, comment='#', sep='\t')
    df_pos = pd.read_csv(POSITIVES_SSSOM_PATH, comment='#', sep='\t')

    def get_mappings_for_pair(df, ns1, ns2, predicate='skos:exactMatch'):
        mappings = set()
        df12 = df[df['subject_id'].str.startswith(ns1) &
                  df['object_id'].str.startswith(ns2) &
                  (df['predicate_id'] == predicate)]
        for _, row in df12.iterrows():
            mappings.add((row['subject_id'], row['object_id']))
        df21 = df[df['subject_id'].str.startswith(ns2) &
                  df['object_id'].str.startswith(ns1) &
                  (df['predicate_id'] == predicate)]
        for _, row in df21.iterrows():
            mappings.add((row['object_id'], row['subject_id']))
        return mappings

    biomappings_pairs = (
        get_mappings_for_pair(df_pred, 'mondo', 'mesh')
        | get_mappings_for_pair(df_pos, 'mondo', 'mesh')
    )

    novel_no_biomappings = [
        m for m in novel_mappings
        if (m['mondo_curie'], m['mesh_curie']) not in biomappings_pairs
    ]
    print(f"Novel after excluding Biomappings: {len(novel_no_biomappings)}")

    # Flag HLBS-relevant mappings
    hlbs_mesh_codes = {
        'C14',      # Cardiovascular Diseases
        'C08',      # Respiratory Tract Diseases
        'C15.378',  # Hematologic Diseases
        'C10.886',  # Sleep Disorders
    }
    hlbs_mesh_ids = set()
    for m in novel_no_biomappings:
        for code in hlbs_mesh_codes:
            if mesh_client.has_tree_prefix(m['mesh_id'], code):
                hlbs_mesh_ids.add(m['mesh_id'])
                break

    hlbs_count = sum(1 for m in novel_no_biomappings
                     if m['mesh_id'] in hlbs_mesh_ids)
    print(f"HLBS-relevant novel mappings: {hlbs_count}")

    # Export as SSSOM TSV
    rows = []
    for m in sorted(novel_no_biomappings, key=lambda x: x['mondo_curie']):
        comment = 'HLBS-relevant' if m['mesh_id'] in hlbs_mesh_ids else ''
        rows.append({
            'subject_id': m['mondo_curie'],
            'subject_label': m['mondo_label'],
            'predicate_id': 'skos:exactMatch',
            'object_id': m['mesh_curie'],
            'object_label': m['mesh_label'],
            'mapping_justification': 'semapv:SemanticSimilarityThresholdMatching',
            'confidence': m['confidence'],
            'mapping_tool': 'logmap',
            'comment': comment,
        })

    df_out = pd.DataFrame(rows)
    output_path = os.path.join(os.path.dirname(__file__),
                               'logmap_mondo_mesh_predictions.sssom.tsv')
    with open(output_path, 'w') as f:
        f.write('#curie_map:\n')
        f.write('#  mondo: http://purl.obolibrary.org/obo/MONDO_\n')
        f.write('#  mesh: https://meshb.nlm.nih.gov/record/ui?ui=\n')
        f.write('#  skos: http://www.w3.org/2004/02/skos/core#\n')
        f.write('#  semapv: https://w3id.org/semapv/vocab/\n')
        f.write('#mapping_set_id: logmap_mondo_mesh_predictions\n')
        f.write('#mapping_tool: logmap\n')
    df_out.to_csv(output_path, sep='\t', index=False, mode='a')
    print(f"Wrote {len(df_out)} mappings to {output_path}")
