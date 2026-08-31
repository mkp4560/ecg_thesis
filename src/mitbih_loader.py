import os
import numpy as np
import wfdb


# =============================================================================
# Configuration
# =============================================================================

# List of all MIT-BIH records (from PhysioNet RECORDS file)
MITBIH_RECORDS = [
    '100', '101', '102', '103', '104',
    '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115',
    '116', '117', '118', '119', '121',
    '122', '123', '124',
    '200', '201', '202', '203', '205',
    '207', '208', '209', '210', '212',
    '213', '214', '215', '217', '219',
    '220', '221', '222', '223', '228',
    '230', '231', '232', '233', '234',
]


def get_mitbih_base_path():
    """
    Base path for MIT-BIH data: one level above this file, in data/mitdb.
    Adjust here if your folder structure changes.
    """
    return os.path.join(os.path.dirname(__file__), '..', 'data', 'mitdb')


def get_mitbih_record_path(record_name: str) -> str:
    """
    Full path (without extension) for a given MIT-BIH record.
    """
    return os.path.join(get_mitbih_base_path(), record_name)


# =============================================================================
# Single-record loading and preprocessing
# =============================================================================

def load_mitbih_record(record_name: str = '100'):
    """
    Load one MIT-BIH record and its annotations via WFDB.

    Returns:
        record  : wfdb.io.record.Record
        annotation : wfdb.io.annotation.Annotation
    """
    record_path = get_mitbih_record_path(record_name)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')

    print(f"Record {record_name}: fs={record.fs}, len={record.sig_len}, "
          f"channels={record.sig_name}, first symbols={annotation.symbol[:10]}")
    return record, annotation


def make_beat_segments(record,
                       annotation,
                       window_size: int = 200,
                       pre_samples: int = 80):
    """
    Create fixed-length segments around each annotated beat.

    Args:
        record:      WFDB Record object.
        annotation:  WFDB Annotation object.
        window_size: total segment length (in samples).
        pre_samples: samples before the annotation index.

    Returns:
        segments: np.ndarray [num_beats, window_size]
        symbols:  list[str] length num_beats
    """
    signal = record.p_signal[:, 0]   # use first channel (e.g. MLII)
    samples = annotation.sample
    symbols = annotation.symbol

    segments = []
    seg_symbols = []

    for idx, sym in zip(samples, symbols):
        start = idx - pre_samples
        end = start + window_size

        # Skip segments that would go out of bounds
        if start < 0 or end > len(signal):
            continue

        seg = signal[start:end]
        segments.append(seg)
        seg_symbols.append(sym)

    segments = np.array(segments, dtype=np.float32)
    return segments, seg_symbols


def map_symbols_to_aami_labels(symbols):
    """
    Map WFDB beat symbols to simplified AAMI classes:

        0: N (normal)        -> {N, L, R, e, j}
        1: S (supraventric)  -> {A, a, J, S}
        2: V (ventricular)   -> {V, E}
        3: F (fusion)        -> {F}
        4: Q (unknown/other) -> everything else
    """
    n_set = {'N', 'L', 'R', 'e', 'j'}
    s_set = {'A', 'a', 'J', 'S'}
    v_set = {'V', 'E'}
    f_set = {'F'}

    labels = []
    for sym in symbols:
        if sym in n_set:
            labels.append(0)
        elif sym in s_set:
            labels.append(1)
        elif sym in v_set:
            labels.append(2)
        elif sym in f_set:
            labels.append(3)
        else:
            labels.append(4)
    return np.array(labels, dtype=np.int64)


def load_segments_labels_for_record(record_name: str = '100',
                                    window_size: int = 200,
                                    pre_samples: int = 80):
    """
    Convenience function: from one record, get segments and AAMI labels.

    Returns:
        segments: np.ndarray [N, window_size]
        labels:   np.ndarray [N]
    """
    record, annotation = load_mitbih_record(record_name)
    segments, symbols = make_beat_segments(record, annotation,
                                           window_size=window_size,
                                           pre_samples=pre_samples)
    labels = map_symbols_to_aami_labels(symbols)
    return segments, labels


# =============================================================================
# Multi-record (full-dataset) loading
# =============================================================================

def load_all_segments_labels(window_size: int = 200,
                             pre_samples: int = 80,
                             record_list=None):
    """
    Load beat-centered segments and AAMI labels from ALL specified MIT-BIH records.

    Args:
        window_size: segment length (in samples).
        pre_samples: samples before annotation index.
        record_list: list of record names (strings). If None, use MITBIH_RECORDS.

    Returns:
        segments_all: np.ndarray [N_total, window_size]
        labels_all:   np.ndarray [N_total]
        record_ids:   np.ndarray [N_total] (record name for each beat)
    """
    if record_list is None:
        record_list = MITBIH_RECORDS

    all_segments = []
    all_labels = []
    all_record_ids = []

    for rec in record_list:
        print(f"\n=== Processing record {rec} ===")
        record, annotation = load_mitbih_record(rec)
        segs, symbols = make_beat_segments(record, annotation,
                                           window_size=window_size,
                                           pre_samples=pre_samples)
        lbls = map_symbols_to_aami_labels(symbols)

        uniq, cnts = np.unique(lbls, return_counts=True)
        print(f"  segments: {segs.shape[0]}, labels (id,count): {list(zip(uniq, cnts))}")

        all_segments.append(segs)
        all_labels.append(lbls)
        all_record_ids.append(np.array([rec] * segs.shape[0]))

    segments_all = np.concatenate(all_segments, axis=0)
    labels_all = np.concatenate(all_labels, axis=0)
    record_ids_all = np.concatenate(all_record_ids, axis=0)

    print("\n=== Combined dataset statistics ===")
    uniq_all, cnts_all = np.unique(labels_all, return_counts=True)
    print("Total segments:", segments_all.shape[0])
    print("Window size:", segments_all.shape[1])
    print("Label distribution (id,count):", list(zip(uniq_all, cnts_all)))

    return segments_all, labels_all, record_ids_all


# =============================================================================
# Simple self-test when run directly
# =============================================================================

if __name__ == "__main__":
    # Quick sanity check on a single record
    print("Base MIT-BIH path:", get_mitbih_base_path())
    segs_100, lbls_100 = load_segments_labels_for_record('100', window_size=200, pre_samples=80)
    print("\nRecord 100 -> segments:", segs_100.shape, "unique labels:", np.unique(lbls_100))

    # Optional: test loading all records (may take some time)
    # Uncomment to run full-dataset check
    # segments_all, labels_all, record_ids_all = load_all_segments_labels(
    #     window_size=200,
    #     pre_samples=80,
    #     record_list=MITBIH_RECORDS
    # )
    # print("All records -> segments:", segments_all.shape,
    #       "unique labels:", np.unique(labels_all, return_counts=True))