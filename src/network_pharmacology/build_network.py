"""
Multilayer network construction and biomarker (hub-gene) identification
(manuscript Methods 2.6 / Results 3.5).

Pipeline:
  1. Build a PPI network restricted to DEGs (from `differential_expression.py`)
     using a reference interaction database (e.g. STRING, config-supplied
     confidence threshold).
  2. Detect communities via the Louvain algorithm (`python-louvain`).
  3. Rank genes by centrality: PageRank and betweenness centrality
     (both computed; manuscript Table 4/5 report both).
  4. Cross-reference top-ranked hub genes against DisGeNET for known
     cardiovascular-disease gene-disease associations.

Note (Discussion, Limitations): as stated in the manuscript, high network
centrality identifies statistically prioritized candidate biomarkers, not
experimentally validated ones -- centrality alone does not establish
causal or clinical biomarker status. This module's ranking output should
be read accordingly, and is downstream input to the retrospective
MIMIC-IV cross-reference in `clinical_validation.psm_cox`, not a
replacement for wet-lab or clinical validation.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import networkx as nx
import pandas as pd
import yaml

try:
    import community as community_louvain  # python-louvain
except ImportError:  # pragma: no cover
    community_louvain = None

log = logging.getLogger(__name__)


@dataclass
class NetworkConfig:
    string_confidence_threshold: float = 0.7
    top_n_hubs: int = 25

    @classmethod
    def from_yaml(cls, path: str) -> "NetworkConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def build_ppi_subnetwork(deg_genes: list[str], string_edges: pd.DataFrame, config: NetworkConfig) -> nx.Graph:
    """`string_edges`: DataFrame with columns {protein1, protein2, combined_score
    (0-1000, STRING convention)}, pre-downloaded per README Data access."""
    edges = string_edges[
        (string_edges["protein1"].isin(deg_genes))
        & (string_edges["protein2"].isin(deg_genes))
        & (string_edges["combined_score"] >= config.string_confidence_threshold * 1000)
    ]
    G = nx.from_pandas_edgelist(edges, "protein1", "protein2", edge_attr="combined_score")
    log.info("Built PPI subnetwork: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


def detect_communities(G: nx.Graph) -> dict:
    if community_louvain is None:
        raise ImportError("python-louvain is required: pip install python-louvain")
    return community_louvain.best_partition(G)


def rank_hub_genes(G: nx.Graph, config: NetworkConfig) -> pd.DataFrame:
    pagerank = nx.pagerank(G)
    betweenness = nx.betweenness_centrality(G)
    degree = dict(G.degree())

    df = pd.DataFrame(
        {
            "gene": list(G.nodes()),
            "pagerank": [pagerank[n] for n in G.nodes()],
            "betweenness_centrality": [betweenness[n] for n in G.nodes()],
            "degree": [degree[n] for n in G.nodes()],
        }
    )
    df = df.sort_values("pagerank", ascending=False).reset_index(drop=True)
    return df.head(config.top_n_hubs)


def cross_reference_disgenet(hub_genes: pd.DataFrame, disgenet_df: pd.DataFrame, disease_keywords: list[str]) -> pd.DataFrame:
    """`disgenet_df`: DisGeNET gene-disease association export (pre-downloaded;
    DisGeNET requires its own account/license -- see README). Flags each hub
    gene with whether it has a prior DisGeNET association to any of
    `disease_keywords` (e.g. ['heart failure', 'cardiomyopathy'])."""
    pattern = "|".join(disease_keywords)
    matches = disgenet_df[disgenet_df["disease_name"].str.contains(pattern, case=False, na=False)]
    known_genes = set(matches["gene_symbol"])

    hub_genes = hub_genes.copy()
    hub_genes["prior_disgenet_association"] = hub_genes["gene"].isin(known_genes)
    return hub_genes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--deg-results", help="Output of differential_expression.py")
    parser.add_argument("--string-edges", help="Pre-downloaded STRING interaction file")
    parser.add_argument("--disgenet", help="Pre-downloaded DisGeNET gene-disease association file")
    parser.add_argument("--output", default="data_processed/hub_genes.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = NetworkConfig.from_yaml(args.config)

    if not (args.deg_results and args.string_edges):
        log.warning(
            "This step requires --deg-results (from differential_expression.py) and "
            "--string-edges (a pre-downloaded STRING export; see README Data access). "
            "Neither is bundled with this repository."
        )
        return

    deg_df = pd.read_csv(args.deg_results)
    deg_genes = deg_df.loc[deg_df["significant"], "gene"].tolist()
    string_edges = pd.read_csv(args.string_edges)

    G = build_ppi_subnetwork(deg_genes, string_edges, config)
    hub_genes = rank_hub_genes(G, config)

    if args.disgenet:
        disgenet_df = pd.read_csv(args.disgenet)
        hub_genes = cross_reference_disgenet(hub_genes, disgenet_df, ["heart failure", "cardiomyopathy", "cardiovascular"])

    hub_genes.to_csv(args.output, index=False)
    log.info("Wrote top %d hub genes to %s", len(hub_genes), args.output)


if __name__ == "__main__":
    main()
