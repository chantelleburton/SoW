"""
Identify ensemble members that are complete (63 consecutive monthly files,
Nov 2019 - Jan 2025, with no gaps) in BOTH historicalExt and historicalNatExt,
and export the matching member IDs to a CSV.

Checks all 525 members (r001-r105 x p1-p5) so members with neither directory
are also tracked.
"""

import os
import re
import pandas as pd

HIST_DIR = (
    '/data/scratch/andrew.hartley/impactstoolbox/Data/attribution_ensemble/'
    'Fire-Weather/FWI/HadGEM3-A-N216/historicalExt'
)
HISTNAT_DIR = (
    '/data/scratch/andrew.hartley/impactstoolbox/Data/attribution_ensemble/'
    'Fire-Weather/FWI/HadGEM3-A-N216/historicalNatExt'
)
OUTPUT_CSV = '/data/scratch/bob.potts/sowf/matching_complete_members.csv'

N_RUNS = 105
N_PHYSICS = 5

# Expected 63 consecutive monthly date stamps for a complete member
EXPECTED_DATES = [
    '20191101-20191201', '20191201-20200101',
    '20200101-20200201', '20200201-20200301', '20200301-20200401',
    '20200401-20200501', '20200501-20200601', '20200601-20200701',
    '20200701-20200801', '20200801-20200901', '20200901-20201001',
    '20201001-20201101', '20201101-20201201', '20201201-20210101',
    '20210101-20210201', '20210201-20210301', '20210301-20210401',
    '20210401-20210501', '20210501-20210601', '20210601-20210701',
    '20210701-20210801', '20210801-20210901', '20210901-20211001',
    '20211001-20211101', '20211101-20211201', '20211201-20220101',
    '20220101-20220201', '20220201-20220301', '20220301-20220401',
    '20220401-20220501', '20220501-20220601', '20220601-20220701',
    '20220701-20220801', '20220801-20220901', '20220901-20221001',
    '20221001-20221101', '20221101-20221201', '20221201-20230101',
    '20230101-20230201', '20230201-20230301', '20230301-20230401',
    '20230401-20230501', '20230501-20230601', '20230601-20230701',
    '20230701-20230801', '20230801-20230901', '20230901-20231001',
    '20231001-20231101', '20231101-20231201', '20231201-20240101',
    '20240101-20240201', '20240201-20240301', '20240301-20240401',
    '20240401-20240501', '20240501-20240601', '20240601-20240701',
    '20240701-20240801', '20240801-20240901', '20240901-20241001',
    '20241001-20241101', '20241101-20241201', '20241201-20250101',
    '20250101-20250201',
]
EXPECTED_SET = set(EXPECTED_DATES)
N_EXPECTED = len(EXPECTED_SET)
DATE_RE = re.compile(r'(\d{8}-\d{8})\.nc$')


def all_member_ids():
    """Generate all 525 member IDs: r001i1p1 ... r105i1p5."""
    for r in range(1, N_RUNS + 1):
        for p in range(1, N_PHYSICS + 1):
            yield f'r{r:03d}i1p{p}'


def count_matching_files(member_path):
    """Return the number of expected monthly files present in a member directory."""
    if not os.path.isdir(member_path):
        return 0
    stamps = set()
    for fname in os.listdir(member_path):
        m = DATE_RE.search(fname)
        if m:
            stamps.add(m.group(1))
    return len(stamps & EXPECTED_SET)


def get_both_complete_member_ids():
    """
    Return a sorted list of member IDs that are complete (all 63 expected files)
    in BOTH historicalExt and historicalNatExt.
    """
    both = []
    for member in all_member_ids():
        hist_path = os.path.join(HIST_DIR, member)
        histnat_path = os.path.join(HISTNAT_DIR, member)
        if (count_matching_files(hist_path) == N_EXPECTED and
                count_matching_files(histnat_path) == N_EXPECTED):
            both.append(member)
    return sorted(both)


def get_complete_member_dirs(run_dir):
    """
    Return a sorted list of full directory paths for every member in `run_dir`
    that has all 63 expected monthly files present.

    Suitable for direct use as the `member_dirs` list in the bias-correction
    script, replacing the CSV-based lookup.
    """
    complete = []
    for member in all_member_ids():
        member_path = os.path.join(run_dir, member)
        if count_matching_files(member_path) == N_EXPECTED:
            complete.append(member_path)
    return sorted(complete)


def main():
    rows = []
    for member in all_member_ids():
        hist_path = os.path.join(HIST_DIR, member)
        histnat_path = os.path.join(HISTNAT_DIR, member)

        hist_exists = os.path.isdir(hist_path)
        histnat_exists = os.path.isdir(histnat_path)

        hist_count = count_matching_files(hist_path) if hist_exists else 0
        histnat_count = count_matching_files(histnat_path) if histnat_exists else 0

        hist_complete = hist_count == N_EXPECTED
        histnat_complete = histnat_count == N_EXPECTED

        rows.append({
            'member': member,
            'historicalExt_exists': hist_exists,
            'historicalNatExt_exists': histnat_exists,
            'historicalExt_file_count': hist_count,
            'historicalNatExt_file_count': histnat_count,
            'historicalExt_complete': hist_complete,
            'historicalNatExt_complete': histnat_complete,
            'both_complete': hist_complete and histnat_complete,
            'neither_exists': not hist_exists and not histnat_exists,
        })

    df = pd.DataFrame(rows)
    matched = df[df['both_complete']]
    neither = df[df['neither_exists']]

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Total members checked:       {len(df)} (={N_RUNS}x{N_PHYSICS})")
    print(f"Exist in historicalExt:      {df['historicalExt_exists'].sum()}")
    print(f"Exist in historicalNatExt:   {df['historicalNatExt_exists'].sum()}")
    print(f"Complete in historicalExt:    {df['historicalExt_complete'].sum()}")
    print(f"Complete in historicalNatExt: {df['historicalNatExt_complete'].sum()}")
    print(f"Complete in BOTH:            {len(matched)}")
    print(f"Neither directory exists:    {len(neither)}")
    print(f"\nMatching members ({len(matched)}):")
    for m in matched['member']:
        print(f"  {m}")
    print(f"\nFull results saved to: {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
