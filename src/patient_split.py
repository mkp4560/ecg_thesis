import numpy as np

from mitbih_loader import load_segments_labels_for_record

# =============================================================================
# Inter-patient (DS1/DS2) split, per de Chazal, O'Dwyer & Reilly,
# "Automatic Classification of Heartbeats Using ECG Morphology and Heartbeat
# Interval Features," IEEE Trans. Biomed. Eng., 51(7), 2004.
#
# Records 102, 104, 107, 217 are excluded (predominantly paced beats).
# Note records 201/202 originate from the same patient but are split across
# DS1/DS2 in the original protocol; this is preserved here to match the
# standard benchmark rather than "fixing" it.
# =============================================================================

DS1_RECORDS = [
    '101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
    '122', '124', '201', '203', '205', '207', '208', '209', '215', '220',
    '223', '230',
]

DS2_RECORDS = [
    '100', '103', '105', '111', '113', '117', '121', '123', '200', '202',
    '210', '212', '213', '214', '219', '221', '222', '228', '231', '232',
    '233', '234',
]

EXCLUDED_PACED_RECORDS = ['102', '104', '107', '217']


def _load_records(record_list, window_size, pre_samples):
    all_segments = []
    all_labels = []
    all_record_ids = []

    for rec in record_list:
        segs, lbls = load_segments_labels_for_record(
            rec, window_size=window_size, pre_samples=pre_samples
        )
        all_segments.append(segs)
        all_labels.append(lbls)
        all_record_ids.append(np.array([rec] * segs.shape[0]))

    segments = np.concatenate(all_segments, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    record_ids = np.concatenate(all_record_ids, axis=0)
    return segments, labels, record_ids


def load_ds1_ds2_segments_labels(window_size: int = 200, pre_samples: int = 80):
    """
    Load the standard inter-patient MIT-BIH split.

    Returns:
        x_train, y_train, train_record_ids : DS1 (training patients)
        x_test,  y_test,  test_record_ids  : DS2 (held-out patients)
    """
    print("\n=== Loading DS1 (inter-patient train) ===")
    x_train, y_train, train_record_ids = _load_records(
        DS1_RECORDS, window_size, pre_samples
    )
    print("DS1 segments:", x_train.shape, "label counts:",
          list(zip(*np.unique(y_train, return_counts=True))))

    print("\n=== Loading DS2 (inter-patient test) ===")
    x_test, y_test, test_record_ids = _load_records(
        DS2_RECORDS, window_size, pre_samples
    )
    print("DS2 segments:", x_test.shape, "label counts:",
          list(zip(*np.unique(y_test, return_counts=True))))

    return x_train, y_train, train_record_ids, x_test, y_test, test_record_ids
