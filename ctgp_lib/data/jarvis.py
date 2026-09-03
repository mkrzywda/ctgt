"""Auto-generated from the CTGP research monolith — do not edit by hand."""

import io
import json
import os
import time
import zipfile
import urllib.request
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from ctgp_lib.config import *
from ctgp_lib.runtime import logger, seed, JARVIS_LOADER_CACHE
from ctgp_lib.device import get_device

def _to_float(value):
    """Convert various value types to float. Handles dicts, arrays, etc."""
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, str):
        txt = value.strip().lower()
        if txt in {'', 'none', 'nan', 'na', 'n/a'}:
            return None
        try:
            return float(txt)
        except ValueError:
            return None
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float).ravel()
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return None
        return float(np.max(np.abs(arr)))
    if isinstance(value, dict):
        for key in ['value', 'magmom', 'magmom_outcar', 'total_magnetization']:
            if key in value:
                out = _to_float(value[key])
                if out is not None:
                    return out
        for val in value.values():
            out = _to_float(val)
            if out is not None:
                return out
        return None
    return None


def _class_dist_string(y):
    y = np.asarray(y, dtype=int)
    if y.size == 0:
        return "empty"
    vals, cnts = np.unique(y, return_counts=True)
    return ", ".join(f"{int(v)}:{int(c)} ({(c / y.size):.3f})" for v, c in zip(vals, cnts))


ELEMENT_TO_Z = {}
PERIODIC_TABLE = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
]
ELEMENT_TO_Z = {el: i + 1 for i, el in enumerate(PERIODIC_TABLE)}

# ---------------------------------------------------------------------------
# Graph representation levels
# Level 0 : x = [Z]                 (1 feature, current behaviour)
# Level 1 : x = cgcnn[92]           (ALIGNN-style atomic embeddings)
# Level 2 : x = cgcnn[92] + RBF(d)  (+ 10-Gaussian radial basis of bond dists)
# Level 3 : x = cgcnn[92] + RBF(d) + angle_stats[4]  (bond-angle approximation)
# ---------------------------------------------------------------------------

_CGCNN_FEAT_LOOKUP = None   # np.ndarray shape [max_Z+2, 92], lazy-initialised


def _build_cgcnn_lookup():
    """Build (once) a numpy lookup table: atomic_number -> cgcnn 92-dim vector."""
    global _CGCNN_FEAT_LOOKUP
    if _CGCNN_FEAT_LOOKUP is not None:
        return
    from jarvis.core.specie import chem_data, get_node_attributes
    max_z = max(v["Z"] for v in chem_data.values())
    template = get_node_attributes("C", atom_features="cgcnn")
    feat_dim = len(template)
    lut = np.zeros((max_z + 2, feat_dim), dtype=np.float32)
    for element, v in chem_data.items():
        z = v["Z"]
        x = get_node_attributes(element, atom_features="cgcnn")
        if x is not None:
            lut[z] = np.array(x, dtype=np.float32)
    _CGCNN_FEAT_LOOKUP = lut


# 10-Gaussian RBF centres uniformly covering [0, 9] Å, width σ=0.5 Å
_RBF_CENTERS = np.linspace(0.5, 8.5, 10).astype(np.float32)
_RBF_WIDTH   = 0.5


def _rbf_encode(distances: np.ndarray) -> np.ndarray:
    """Return per-distance RBF vectors, averaged to a single [10]-dim signature."""
    # distances: (k,)  → rbf: (k, 10) → mean → (10,)
    diff = distances[:, None] - _RBF_CENTERS[None, :]   # (k, 10)
    rbf  = np.exp(-0.5 * (diff / _RBF_WIDTH) ** 2)      # (k, 10)
    return rbf.mean(axis=0).astype(np.float32)


def _safe_neighbor_data(neighbor):
    """Return (j_index, distance, image_array) from a JARVIS neighbor object."""
    if isinstance(neighbor, (list, tuple)):
        if len(neighbor) >= 4:
            return int(neighbor[1]), float(neighbor[2]), np.array(neighbor[3], dtype=np.float32)
        if len(neighbor) >= 3:
            return int(neighbor[1]), float(neighbor[2]), np.zeros(3, dtype=np.float32)
        return None, None, None
    idx  = getattr(neighbor, 'index', None)
    dist = getattr(neighbor, 'distance', None)
    img  = getattr(neighbor, 'image', np.zeros(3))
    if idx is None or dist is None:
        return None, None, None
    return int(idx), float(dist), np.array(img, dtype=np.float32)


def _compute_repr_features(atoms, zs, all_nbrs, max_neighbors, repr_level):
    """Return per-atom feature arrays for level 2/3 (RBF distances + angle stats).

    Returns:
        rbf_feats   : np.ndarray shape [n, 10]  (zeros for level < 2)
        angle_feats : np.ndarray shape [n, 4]   (zeros for level < 3)
    """
    n = len(zs)
    rbf_feats   = np.zeros((n, 10), dtype=np.float32)
    angle_feats = np.zeros((n,  4), dtype=np.float32)

    if repr_level < 2:
        return rbf_feats, angle_feats

    frac  = np.array(atoms.frac_coords,  dtype=np.float64)
    latt  = atoms.lattice               # jarvis Lattice object

    for i, nbrs in enumerate(all_nbrs):
        parsed = []
        vecs   = []
        for nb in nbrs:
            j, d, img = _safe_neighbor_data(nb)
            if j is None or j < 0 or j >= n:
                continue
            parsed.append((j, d, img))

        parsed.sort(key=lambda t: t[1])
        parsed = parsed[:max_neighbors]
        if not parsed:
            continue

        dists = np.array([d for _, d, _ in parsed], dtype=np.float32)
        rbf_feats[i] = _rbf_encode(dists)

        if repr_level >= 3 and len(parsed) >= 2:
            for j, d, img in parsed:
                # Periodic-image-correct displacement vector  i → j
                frac_j = frac[j] + img.astype(np.float64)
                dv     = latt.cart_coords(frac_j - frac[i])   # (3,) array
                norm   = np.linalg.norm(dv)
                if norm > 1e-6:
                    vecs.append(dv / norm)

            cosines = []
            for a in range(len(vecs)):
                for b in range(a + 1, len(vecs)):
                    cosines.append(float(np.clip(np.dot(vecs[a], vecs[b]), -1.0, 1.0)))

            if cosines:
                c = np.array(cosines, dtype=np.float32)
                angle_feats[i] = [c.mean(), c.std(), c.min(), c.max()]

    return rbf_feats, angle_feats


def _jarvis_cls_label(entry, class_target, threshold_override=None):
    """Extract binary classification label from JARVIS entry.
    
    Supports 10 classification tasks from JARVIS-DFT ALIGNN paper:
    1. metal_opt: bandgap (OPT functional)
    2. metal_mbj: bandgap (MBJ functional)
    3. magnetic: magnetic moment
    4. stable: formation energy (ehull)
    5. slme: Spectroscopic Limited Maximum Efficiency
    6. spillage: band structure spillage
    7. seebeck_n: n-type Seebeck coefficient (low/high)
    8. seebeck_p: p-type Seebeck coefficient (low/high)
    9. powerfactor_n: n-type power factor (low/high)
    10. powerfactor_p: p-type power factor (low/high)
    """
    
    # METAL CLASSIFICATION (bandgap-based)
    if class_target == 'metal_opt':
        key = 'optb88vdw_bandgap'
        threshold = 0.01 if threshold_override is None else threshold_override
        val = _to_float(entry.get(key))
        if val is None:
            return None
        return int(val <= threshold)  # 1=metal, 0=non-metal
    
    if class_target == 'metal_mbj':
        key = 'mbj_bandgap'
        threshold = 0.01 if threshold_override is None else threshold_override
        val = _to_float(entry.get(key))
        if val is None:
            return None
        return int(val <= threshold)  # 1=metal, 0=non-metal
    
    # MAGNETIC CLASSIFICATION
    if class_target == 'magnetic':
        key = 'magmom_outcar'
        threshold = 0.05 if threshold_override is None else threshold_override
        val = _to_float(entry.get(key))
        if val is None:
            return None
        return int(abs(val) >= threshold)  # 1=magnetic, 0=non-magnetic
    
    # STABILITY CLASSIFICATION
    if class_target == 'stable':
        # Stable = formation energy (ehull) < threshold
        key = 'ehull'
        threshold = 0.1 if threshold_override is None else threshold_override  # 0.1 eV
        val = _to_float(entry.get(key))
        if val is None:
            return None
        return int(val < threshold)  # 1=stable, 0=unstable
    
    # OPTICAL PROPERTIES (SLME)
    if class_target == 'slme':
        # High/Low SLME efficiency
        key = 'slme'
        threshold = 10.0 if threshold_override is None else threshold_override  # 10%
        val = _to_float(entry.get(key))
        if val is None:
            return None
        return int(val >= threshold)  # 1=high SLME, 0=low SLME
    
    # BAND STRUCTURE SPILLAGE
    if class_target == 'spillage':
        # High/Low spillage
        key = 'spillage'
        threshold = 0.1 if threshold_override is None else threshold_override
        val = _to_float(entry.get(key))
        if val is None:
            return None
        return int(val >= threshold)  # 1=high spillage, 0=low spillage
    
    # THERMOELECTRIC: N-TYPE SEEBECK
    if class_target == 'seebeck_n':
        # High/Low n-type Seebeck coefficient
        # Try multiple column names for compatibility with different JARVIS versions
        key = None
        for possible_key in ['seebeck_doped_n', 'n-Seebeck']:
            if _to_float(entry.get(possible_key)) is not None:
                key = possible_key
                break
        if key is None:
            return None
        threshold = -100.0 if threshold_override is None else threshold_override  # µVK⁻¹
        val = _to_float(entry.get(key))
        return int(val <= threshold)  # 1=high n-Seebeck, 0=low n-Seebeck
    
    # THERMOELECTRIC: P-TYPE SEEBECK
    if class_target == 'seebeck_p':
        # High/Low p-type Seebeck coefficient
        # Try multiple column names for compatibility with different JARVIS versions
        key = None
        for possible_key in ['seebeck_doped_p', 'p-Seebeck']:
            if _to_float(entry.get(possible_key)) is not None:
                key = possible_key
                break
        if key is None:
            return None
        threshold = 100.0 if threshold_override is None else threshold_override  # µVK⁻¹
        val = _to_float(entry.get(key))
        return int(val >= threshold)  # 1=high p-Seebeck, 0=low p-Seebeck
    
    # THERMOELECTRIC: N-TYPE POWER FACTOR
    if class_target == 'powerfactor_n':
        # High/Low n-type power factor
        # Try multiple column names for compatibility with different JARVIS versions
        key = None
        for possible_key in ['powerfactor_doped_n', 'n-powerfact']:
            if _to_float(entry.get(possible_key)) is not None:
                key = possible_key
                break
        if key is None:
            return None
        threshold = 1000.0 if threshold_override is None else threshold_override  # µW(mK²)⁻¹ (from ALIGNN paper)
        val = _to_float(entry.get(key))
        return int(val >= threshold)  # 1=high n-PF, 0=low n-PF
    
    # THERMOELECTRIC: P-TYPE POWER FACTOR
    if class_target == 'powerfactor_p':
        # High/Low p-type power factor
        # Try multiple column names for compatibility with different JARVIS versions
        key = None
        for possible_key in ['powerfactor_doped_p', 'p-powerfact']:
            if _to_float(entry.get(possible_key)) is not None:
                key = possible_key
                break
        if key is None:
            return None
        threshold = 1000.0 if threshold_override is None else threshold_override  # µW(mK²)⁻¹ (from ALIGNN paper)
        val = _to_float(entry.get(key))
        return int(val >= threshold)  # 1=high p-PF, 0=low p-PF
    
    raise ValueError(
        f"Unsupported JARVIS classification target: {class_target}. "
        f"Supported: metal_opt, metal_mbj, magnetic, stable, slme, spillage, "
        f"seebeck_n, seebeck_p, powerfactor_n, powerfactor_p"
    )


def _safe_neighbor_index_and_dist(neighbor):
    if isinstance(neighbor, (list, tuple)) and len(neighbor) >= 3:
        return int(neighbor[1]), float(neighbor[2])
    idx = getattr(neighbor, 'index', None)
    dist = getattr(neighbor, 'distance', None)
    if idx is None or dist is None:
        return None, None
    return int(idx), float(dist)


def _jarvis_entry_to_pyg_data(entry, atoms_cls, cutoff=8.0, max_neighbors=12, debug=False, repr_level=0):
    """Convert JARVIS entry to PyG Data object
    
    Handles both old and new JARVIS-tools formats:
    - Old: entry['atoms'] is dict with 'from_dict'
    - New: entry has direct structure info or 'structure' key
    - Modern: entry may have 'structure' field from ASE/other format
    """
    if debug:
        print(f"  Step 1: Check entry type...", flush=True)
    
    if not isinstance(entry, dict):
        if debug:
            print(f"    ✗ entry is {type(entry)}, not dict", flush=True)
        return None
    
    if debug:
        keys = list(entry.keys())
        print(f"  Step 2: Entry has {len(keys)} keys: {keys[:5]}{'...' if len(keys)>5 else ''}", flush=True)
    
    atoms = None
    
    # Try method 1: entry['atoms'] as dict (old format)
    if debug:
        print(f"  Step 3a: Try entry['atoms']...", flush=True)
    atoms_dict = entry.get('atoms')
    if isinstance(atoms_dict, dict):
        try:
            atoms = atoms_cls.from_dict(atoms_dict)
            if debug:
                print(f"    ✓ Successfully loaded atoms from entry['atoms']", flush=True)
        except Exception as e:
            if debug:
                print(f"    ✗ from_dict(atoms) failed: {type(e).__name__}", flush=True)
            atoms = None
    elif debug:
        print(f"    ✗ entry['atoms'] is {type(atoms_dict).__name__ if atoms_dict else 'missing'}", flush=True)
    
    # Try method 2: entry['structure'] 
    if atoms is None:
        if debug:
            print(f"  Step 3b: Try entry['structure']...", flush=True)
        struct = entry.get('structure')
        if isinstance(struct, dict):
            try:
                atoms = atoms_cls.from_dict(struct)
                if debug:
                    print(f"    ✓ Successfully loaded atoms from entry['structure']", flush=True)
            except Exception as e:
                if debug:
                    print(f"    ✗ from_dict(structure) failed: {type(e).__name__}", flush=True)
                atoms = None
        elif debug:
            print(f"    ✗ entry['structure'] is {type(struct).__name__ if struct else 'missing'}", flush=True)
    
    # Try method 3: Check if entry itself _is_ an Atoms object or has necessary properties
    if atoms is None:
        if debug:
            print(f"  Step 3c: Check if entry is Atoms-like...", flush=True)
        try:
            test_elements = getattr(entry, 'elements', None)
            if test_elements:
                atoms = entry
                if debug:
                    print(f"    ✓ Entry is Atoms-like object", flush=True)
        except Exception as e:
            if debug:
                print(f"    ✗ Entry check failed: {e}", flush=True)
            atoms = None
    
    # If all methods fail
    if atoms is None:
        if debug:
            print(f"  FAIL: Could not create Atoms object from any method", flush=True)
        return None

    if debug:
        print(f"  Step 4: Extract elements...", flush=True)
    elements = list(getattr(atoms, 'elements', []))
    if not elements:
        if debug:
            print(f"    ✗ No elements found in atoms", flush=True)
        return None
    
    if debug:
        print(f"    ✓ Found {len(elements)} elements", flush=True)
    
    if debug:
        print(f"  Step 5: Check element symbols...", flush=True)
    zs = [ELEMENT_TO_Z.get(el, 0) for el in elements]
    if any(z == 0 for z in zs):
        if debug:
            invalid = [e for e in elements if ELEMENT_TO_Z.get(e, 0) == 0]
            print(f"    ✗ Unknown elements: {invalid}", flush=True)
        return None
    
    if debug:
        print(f"    ✓ All elements valid", flush=True)

    if debug:
        print(f"  Step 6: Get neighbors...", flush=True)
    try:
        all_nbrs = atoms.get_all_neighbors(r=cutoff)
        if debug:
            print(f"    ✓ get_all_neighbors(r={cutoff}) succeeded", flush=True)
    except Exception as e:
        if debug:
            print(f"    ✗ get_all_neighbors failed: {type(e).__name__}", flush=True)
        return None

    if debug:
        print(f"  Step 7: Build graph edges...", flush=True)
    src, dst = [], []
    num_nodes = len(zs)
    for i, nbrs in enumerate(all_nbrs):
        parsed = []
        for nb in nbrs:
            j, d = _safe_neighbor_index_and_dist(nb)
            if j is None or j < 0 or j >= num_nodes or j == i:
                continue
            parsed.append((j, d))
        parsed.sort(key=lambda t: t[1])
        for j, _ in parsed[:max_neighbors]:
            src.append(i)
            dst.append(j)

    if not src:
        src = list(range(num_nodes))
        dst = list(range(num_nodes))

    if debug:
        print(f"    ✓ Created graph with {len(src)} edges", flush=True)

    # ---- Node feature matrix based on repr_level ----
    if repr_level == 0:
        x = torch.tensor(zs, dtype=torch.float).view(-1, 1)
    else:
        _build_cgcnn_lookup()  # lazy init (no-op after first call)
        cgcnn = _CGCNN_FEAT_LOOKUP[zs]          # (n, 92)
        if repr_level >= 2:
            rbf_f, angle_f = _compute_repr_features(
                atoms, zs, all_nbrs, max_neighbors, repr_level
            )
            if repr_level == 2:
                cgcnn = np.concatenate([cgcnn, rbf_f], axis=1)  # (n, 102)
            else:  # level 3
                cgcnn = np.concatenate([cgcnn, rbf_f, angle_f], axis=1)  # (n, 106)
        x = torch.tensor(cgcnn, dtype=torch.float)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


def download_leaderboard_splits(benchmark_name, cache_dir='.jarvis_lb_cache'):
    """Download pre-determined train/val/test splits from JARVIS Leaderboard GitHub.

    The JSON has the structure::

        {"train": {"JVASP-1": <val>, ...}, "val": {...}, "test": {...}}

    For SinglePropertyClass tasks the values are already binary (0/1).

    Args:
        benchmark_name: leaderboard benchmark name, e.g. 'dft_3d_spillage'
        cache_dir: local directory to cache the unpacked JSON

    Returns:
        dict with 'train', 'val', 'test' keys mapping jid -> int label
    """
    import io
    import zipfile
    import urllib.request

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'{benchmark_name}.json')

    if os.path.exists(cache_path):
        logger.info(f"Using cached leaderboard splits: {cache_path}")
        with open(cache_path) as f:
            raw = json.load(f)
    else:
        url = f'{_LB_BASE_URL}/{benchmark_name}.json.zip'
        logger.info(f"Downloading leaderboard splits from {url}")
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                data = resp.read()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download leaderboard splits for '{benchmark_name}': {exc}\n"
                f"  URL: {url}\n"
                f"  Check your internet connection or provide a manual cache at {cache_path}"
            ) from exc

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            json_names = [n for n in zf.namelist() if n.endswith('.json')]
            if not json_names:
                raise RuntimeError(f"No JSON file found inside {benchmark_name}.json.zip")
            raw = json.loads(zf.read(json_names[0]))

        with open(cache_path, 'w') as f:
            json.dump(raw, f)
        logger.info(f"Cached leaderboard splits to {cache_path}")

    # Normalise: ensure int labels and consistent key names
    splits = {}
    for split_name in ('train', 'val', 'test'):
        entry = raw.get(split_name, {})
        if isinstance(entry, list):
            # Some benchmarks store a list of dicts: [{"id": jid, "target": val}, ...]
            entry = {str(d['id']): d['target'] for d in entry if 'id' in d and 'target' in d}
        splits[split_name] = {str(k): int(float(v)) for k, v in entry.items()}

    total = sum(len(v) for v in splits.values())
    logger.info(
        f"Leaderboard splits loaded: train={len(splits['train'])}, "
        f"val={len(splits['val'])}, test={len(splits['test'])}, total={total}"
    )
    return splits


def get_jarvis_loaders(batch_size=32, jarvis_dataset='dft_3d', class_target='metal_opt',
                       class_threshold=0.01, cutoff=8.0, max_neighbors=12,
                       max_samples=0, log_every=2000, leaderboard_splits=None,
                       repr_level=0):
    """Load JARVIS data with configurable classification target (10 tasks from ALIGNN paper).

    When *leaderboard_splits* is provided (a dict returned by
    :func:`download_leaderboard_splits`), the pre-determined JARVIS Leaderboard
    train/val/test JID sets are used instead of a random split.  *max_samples*
    is ignored in that mode.
    """
    cache_key = (batch_size, jarvis_dataset, class_target, class_threshold, cutoff, max_neighbors,
                 max_samples, repr_level,
                 id(leaderboard_splits) if leaderboard_splits is not None else None)
    if cache_key in JARVIS_LOADER_CACHE:
        logger.info(f"Using cached JARVIS loaders (dataset={jarvis_dataset}, target={class_target})")
        return JARVIS_LOADER_CACHE[cache_key]

    try:
        from jarvis.db.figshare import data as jarvis_data
        from jarvis.core.atoms import Atoms
    except Exception as exc:
        raise ImportError("JARVIS mode requires `jarvis-tools`. Install it before running.") from exc

    rows_all = jarvis_data(jarvis_dataset)

    # ------------------------------------------------------------------
    # LEADERBOARD MODE: use pre-determined JARVIS Leaderboard JID splits
    # ------------------------------------------------------------------
    if leaderboard_splits is not None:
        logger.info(
            f"LEADERBOARD MODE: building graphs for pre-determined JID splits "
            f"(dataset={jarvis_dataset}, target={class_target}, "
            f"cutoff={cutoff}, max_neighbors={max_neighbors})"
        )
        t0 = time.time()

        # Build JID → row lookup from the full dataset
        jid_lookup = {}
        for row in rows_all:
            jid = str(row.get('jid', ''))
            if jid:
                jid_lookup[jid] = row

        def _build_split_graphs(split_dict, split_name):
            split_graphs, skipped, not_found = [], 0, 0
            for jid, label_val in split_dict.items():
                row = jid_lookup.get(jid)
                if row is None:
                    not_found += 1
                    continue
                g = _jarvis_entry_to_pyg_data(row, Atoms, cutoff=cutoff,
                                              max_neighbors=max_neighbors, debug=False,
                                              repr_level=repr_level)
                if g is None:
                    skipped += 1
                    continue
                g.y = torch.tensor([int(label_val)], dtype=torch.long)
                g.jid = jid
                split_graphs.append(g)
            if not_found:
                logger.warning(f"[{split_name}] {not_found} JIDs not found in dataset (dataset may differ)")
            if skipped:
                logger.warning(f"[{split_name}] {skipped} graphs skipped (failed to build)")
            return split_graphs

        tr = _build_split_graphs(leaderboard_splits['train'], 'train')
        vl = _build_split_graphs(leaderboard_splits['val'],   'val')
        ts = _build_split_graphs(leaderboard_splits['test'],  'test')

        for split_name, split_graphs in [('train', tr), ('val', vl), ('test', ts)]:
            if len(split_graphs) < 5:
                raise ValueError(
                    f"Too few usable graphs in '{split_name}' split ({len(split_graphs)}). "
                    "Check that the JARVIS dataset version matches the leaderboard benchmark."
                )

        y_tr = np.array([int(g.y.item()) for g in tr])
        y_vl = np.array([int(g.y.item()) for g in vl])
        y_ts = np.array([int(g.y.item()) for g in ts])
        elapsed = time.time() - t0
        logger.info(
            f"LEADERBOARD splits ready: train={len(tr)}, val={len(vl)}, test={len(ts)}, "
            f"elapsed={elapsed:.1f}s"
        )
        logger.info(
            f"Class distribution: train[{_class_dist_string(y_tr)}] | "
            f"val[{_class_dist_string(y_vl)}] | test[{_class_dist_string(y_ts)}]"
        )

        loaders = (
            DataLoader(tr, batch_size=batch_size, shuffle=True),
            DataLoader(vl, batch_size=batch_size),
            DataLoader(ts, batch_size=batch_size),
            tr[0].num_node_features,
            2,
        )
        JARVIS_LOADER_CACHE[cache_key] = loaders
        return loaders

    # ------------------------------------------------------------------
    # STANDARD MODE: random 80/10/10 split (original behaviour)
    # ------------------------------------------------------------------
    if max_samples > 0 and max_samples < len(rows_all):
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(rows_all), size=max_samples, replace=False)
        rows = [rows_all[i] for i in chosen]
        logger.info(f"Randomly sampled {len(rows)}/{len(rows_all)} rows (seed={seed})")
    else:
        rows = rows_all

    graphs, labels = [], []
    skipped = 0
    skip_reasons = {}  # Count reasons for skipping
    total_rows = len(rows)
    t0 = time.time()
    logger.info(
        f"Preparing JARVIS graphs: dataset={jarvis_dataset}, rows={total_rows}, "
        f"target={class_target}, cutoff={cutoff}, max_neighbors={max_neighbors}"
    )

    for idx, row in enumerate(rows, start=1):
        # DEBUG: Enable debug output for first 10 samples
        debug_this = (idx <= 10)
        if debug_this:
            print(f"\n>>> RECORD {idx}/{total_rows} <<<", flush=True)
        
        graph = _jarvis_entry_to_pyg_data(row, Atoms, cutoff=cutoff, max_neighbors=max_neighbors, debug=debug_this, repr_level=repr_level)
        if graph is None:
            skipped += 1
            reason = row.get('_skip_reason', 'unknown_format')
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            if debug_this:
                print(f"  SKIPPED: {reason}", flush=True)
            continue

        label = _jarvis_cls_label(row, class_target, threshold_override=class_threshold)
        if label is None:
            skipped += 1
            skip_reasons['no_label'] = skip_reasons.get('no_label', 0) + 1
            if debug_this:
                print(f"  SKIPPED: no label", flush=True)
            continue

        graph.y = torch.tensor([int(label)], dtype=torch.long)
        labels.append(int(label))
        graph.jid = str(row.get('jid', 'NA'))
        graphs.append(graph)
        if debug_this:
            print(f"  SUCCESS: label={label}", flush=True)

        if log_every > 0 and (idx % log_every == 0):
            elapsed = time.time() - t0
            logger.info(f"JARVIS progress: {idx}/{total_rows}, kept={len(graphs)}, skipped={skipped}, elapsed={elapsed:.1f}s")
    
    # Log skip reasons summary
    if skip_reasons:
        skip_summary = " | ".join([f"{k}:{v}" for k, v in sorted(skip_reasons.items(), key=lambda x: -x[1])])
        logger.warning(f"SKIP REASONS SUMMARY: {skip_summary}")

    if len(graphs) < 20:
        # Provide helpful diagnostics
        first_row = rows[0] if rows else {}
        has_target = class_target in first_row if isinstance(first_row, dict) else False
        
        diagnostic_msg = f"Too few usable JARVIS samples ({len(graphs)}). Skipped {skipped}."
        if not has_target:
            diagnostic_msg += f"\n  ✗ Target '{class_target}' not found in JARVIS data!"
            diagnostic_msg += f"\n    Available keys: {list(first_row.keys())[:10]}"
            diagnostic_msg += f"\n    Try using: metal_opt, metal_mbj, magnetic, stable, slme, spillage, seebeck_n, seebeck_p, powerfactor_n, powerfactor_p"
        else:
            diagnostic_msg += f"\n  Target '{class_target}' exists, but no samples passed filtering."
            if skip_reasons:
                diagnostic_msg += f"\n  Top failure reasons: {skip_summary[:200]}"
        
        logger.error(diagnostic_msg)
        raise ValueError(diagnostic_msg)

    idx = np.arange(len(graphs))
    y_cls = np.asarray(labels, dtype=int)
    if np.unique(y_cls).size < 2:
        raise ValueError("Classification target has <2 classes")

    tr_idx, ts_idx = train_test_split(idx, test_size=0.1, random_state=seed, stratify=y_cls)
    tr_labels = y_cls[tr_idx]
    tr_idx, vl_idx = train_test_split(tr_idx, test_size=(1.0 / 9.0), random_state=seed, stratify=tr_labels)

    y_tr = y_cls[tr_idx]
    y_vl = y_cls[vl_idx]
    y_ts = y_cls[ts_idx]

    logger.info(
        f"JARVIS class distribution: full[{_class_dist_string(y_cls)}] | "
        f"train[{_class_dist_string(y_tr)}] | val[{_class_dist_string(y_vl)}] | test[{_class_dist_string(y_ts)}]"
    )

    tr = [graphs[i] for i in tr_idx]
    vl = [graphs[i] for i in vl_idx]
    ts = [graphs[i] for i in ts_idx]
    elapsed = time.time() - t0
    logger.info(f"JARVIS ready: total={len(graphs)}, train={len(tr)}, val={len(vl)}, test={len(ts)}, elapsed={elapsed:.1f}s")

    loaders = (
        DataLoader(tr, batch_size=batch_size, shuffle=True),
        DataLoader(vl, batch_size=batch_size),
        DataLoader(ts, batch_size=batch_size),
        tr[0].num_node_features,
        2  # binary classification
    )
    JARVIS_LOADER_CACHE[cache_key] = loaders
    return loaders

