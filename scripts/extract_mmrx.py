"""
Extract recording.edf from .mmrx archives in data/raw/patient_*/.

The .mmrx file is a ZIP archive produced by ResMed ApneaLink Air. It contains
one or more session subfolders, each with a main signal EDF and an Events EDF.
We pick the largest valid signal EDF (skipping discontinuous fragments and the
*_Events.edf metadata files) and copy it to recording.edf next to the .mmrx.

Idempotent: skips patients that already have recording.edf.
"""
import os
import sys
import glob
import shutil
import zipfile
import tempfile

import pyedflib

try:
    import edfio
    HAS_EDFIO = True
except ImportError:
    HAS_EDFIO = False


def find_signal_edfs(extracted_dir):
    candidates = []
    for path in glob.glob(os.path.join(extracted_dir, "**", "*.edf"), recursive=True):
        if path.endswith("_Events.edf"):
            continue
        candidates.append(path)
    return candidates


def is_readable_edf(path):
    try:
        f = pyedflib.EdfReader(path)
        duration = f.getFileDuration()
        f.close()
        return duration > 0
    except Exception:
        return False


def rewrite_edf_plus_d_as_continuous(src, dst):
    """ApneaLink Air writes EDF+D when the recording has gaps (device paused
    and resumed). pyedflib refuses EDF+D. Rebuild as continuous EDF by placing
    each data record at its real-time onset (from the EDF+D annotations) and
    zero-filling the gaps. This preserves the timeline so events.json
    timestamps still align with the signal."""
    if not HAS_EDFIO:
        return False
    import numpy as np
    try:
        e = edfio.read_edf(src)
        rec_dur = e.data_record_duration
        n_rec = e.num_data_records
        anns = e.annotations
        if len(anns) < n_rec:
            return False
        onsets = [anns[i].onset for i in range(n_rec)]
        total_seconds = onsets[-1] + rec_dur

        new_signals = []
        for s in e.signals:
            fs = s.sampling_frequency
            samples_per_record = int(round(rec_dur * fs))
            total_samples = int(round(total_seconds * fs))
            buf = np.zeros(total_samples, dtype=s.data.dtype)
            for i in range(n_rec):
                src_start = i * samples_per_record
                src_end = src_start + samples_per_record
                dst_start = int(round(onsets[i] * fs))
                dst_end = dst_start + samples_per_record
                if dst_end > total_samples or src_end > len(s.data):
                    continue
                buf[dst_start:dst_end] = s.data[src_start:src_end]
            new_signals.append(edfio.EdfSignal(
                data=buf,
                sampling_frequency=fs,
                label=s.label,
                physical_dimension=s.physical_dimension,
                physical_range=(float(s.physical_min), float(s.physical_max)),
                digital_range=(int(s.digital_min), int(s.digital_max)),
            ))
        edfio.Edf(
            signals=new_signals,
            starttime=e.starttime,
            recording=e.recording,
        ).write(dst)
        return is_readable_edf(dst)
    except Exception:
        return False


def extract_one(mmrx_path, target_edf):
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(mmrx_path) as z:
            z.extractall(tmpdir)

        candidates = find_signal_edfs(tmpdir)
        if not candidates:
            return False, "no signal EDF inside archive"

        readable = [(p, os.path.getsize(p)) for p in candidates if is_readable_edf(p)]
        rewrote = False
        if not readable and HAS_EDFIO:
            for p in sorted(candidates, key=os.path.getsize, reverse=True):
                fixed = p + ".rewritten.edf"
                if rewrite_edf_plus_d_as_continuous(p, fixed):
                    readable = [(fixed, os.path.getsize(fixed))]
                    rewrote = True
                    break
        if not readable:
            return False, f"all {len(candidates)} EDFs unreadable (likely EDF+D — install edfio)"

        readable.sort(key=lambda x: x[1], reverse=True)
        chosen, size = readable[0]
        shutil.copy(chosen, target_edf)
        skipped = len(candidates) - len(readable)
        note = f"chose {os.path.basename(chosen)} ({size:,} bytes)"
        if rewrote:
            note += " [EDF+D → EDF+C rewrite]"
        if skipped:
            note += f"; skipped {skipped} unreadable"
        return True, note


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    raw_dir = os.path.join(project_root, "data", "raw")

    patient_dirs = sorted(glob.glob(os.path.join(raw_dir, "patient_*")))
    if not patient_dirs:
        print("No patient_* folders found.")
        return 1

    n_done = n_skip = n_fail = 0
    for p_dir in patient_dirs:
        name = os.path.basename(p_dir)
        edf_path = os.path.join(p_dir, "recording.edf")
        mmrx_path = os.path.join(p_dir, "recording.mmrx")

        if os.path.exists(edf_path):
            print(f"[skip ] {name}: recording.edf already exists")
            n_skip += 1
            continue
        if not os.path.exists(mmrx_path):
            print(f"[fail ] {name}: no recording.edf and no recording.mmrx")
            n_fail += 1
            continue

        ok, note = extract_one(mmrx_path, edf_path)
        if ok:
            print(f"[ok   ] {name}: {note}")
            n_done += 1
        else:
            print(f"[fail ] {name}: {note}")
            n_fail += 1

    print(f"\nDone. extracted={n_done} skipped={n_skip} failed={n_fail}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
