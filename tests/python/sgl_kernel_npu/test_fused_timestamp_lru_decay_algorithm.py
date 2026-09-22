import random
import unittest


def partition_then_merge(stamps, hit_mask, stamp_max):
    """CPU model of the probation kernel's partition + stable merge path."""
    updated = [min(stamp, stamp_max - 1) + 1 for stamp in stamps]
    updated = [
        stamp // 2 if is_hit else stamp
        for stamp, is_hit in zip(updated, hit_mask)
    ]
    non_hits = [
        (stamp, index) for index, (stamp, is_hit) in enumerate(zip(updated, hit_mask))
        if not is_hit
    ]
    hits = [
        (stamp, index) for index, (stamp, is_hit) in enumerate(zip(updated, hit_mask))
        if is_hit
    ]

    merged = []
    non_hit_pos = 0
    hit_pos = 0
    while non_hit_pos < len(non_hits) or hit_pos < len(hits):
        if hit_pos >= len(hits):
            take_non_hit = True
        elif non_hit_pos >= len(non_hits):
            take_non_hit = False
        else:
            non_hit = non_hits[non_hit_pos]
            hit = hits[hit_pos]
            take_non_hit = non_hit[0] > hit[0] or (
                non_hit[0] == hit[0] and non_hit[1] < hit[1]
            )
        if take_non_hit:
            merged.append(non_hits[non_hit_pos])
            non_hit_pos += 1
        else:
            merged.append(hits[hit_pos])
            hit_pos += 1
    return merged


def select_victims_then_merge(stamps, hit_mask, stamp_max, miss_count):
    updated = [min(stamp, stamp_max - 1) + 1 for stamp in stamps]
    updated = [
        stamp // 2 if is_hit else stamp
        for stamp, is_hit in zip(updated, hit_mask)
    ]
    non_hits = [index for index, is_hit in enumerate(hit_mask) if not is_hit]
    victims = non_hits[:miss_count]
    victim_set = set(victims)
    survivors = sorted(
        (
            (stamp, index)
            for index, stamp in enumerate(updated)
            if index not in victim_set
        ),
        key=lambda pair: pair[0],
        reverse=True,
    )
    return victims, survivors


class TestFusedTimestampLruDecayAlgorithm(unittest.TestCase):
    def test_partition_merge_matches_stable_full_sort(self):
        rng = random.Random(20260922)
        for length in (1, 2, 7, 32, 4096):
            for _ in range(20):
                stamp_max = rng.choice((7, 31, (1 << 24) - 1))
                stamps = sorted(
                    (rng.randrange(stamp_max + 1) for _ in range(length)),
                    reverse=True,
                )
                hit_mask = [bool(rng.getrandbits(1)) for _ in range(length)]
                actual = partition_then_merge(stamps, hit_mask, stamp_max)

                updated = [min(stamp, stamp_max - 1) + 1 for stamp in stamps]
                updated = [
                    stamp // 2 if is_hit else stamp
                    for stamp, is_hit in zip(updated, hit_mask)
                ]
                expected = sorted(
                    ((stamp, index) for index, stamp in enumerate(updated)),
                    key=lambda pair: pair[0],
                    reverse=True,
                )
                self.assertEqual(actual, expected)

    def test_odd_hit_stamp_uses_floor_division(self):
        # old=6 ages to 7, then a hit decays to floor(7/2)=3.
        merged = partition_then_merge([6], [True], stamp_max=31)
        self.assertEqual(merged, [(3, 0)])

    def test_saturated_hit_is_halved_after_saturation(self):
        merged = partition_then_merge([7], [True], stamp_max=7)
        self.assertEqual(merged, [(3, 0)])

    def test_repeated_hits_decay_once_per_round(self):
        stamp = 4095
        stamp = partition_then_merge([stamp], [True], stamp_max=(1 << 24) - 1)[0][0]
        stamp = partition_then_merge([stamp], [True], stamp_max=(1 << 24) - 1)[0][0]
        self.assertEqual(stamp, 1024)

    def test_current_hits_are_never_selected_as_victims(self):
        # The oldest entry is a hit that remains relatively old after /2.
        # Victim selection must still use only the non-hit group.
        stamps = [100, 90, 80, 70]
        hit_mask = [True, False, False, False]
        victims, survivors = select_victims_then_merge(
            stamps, hit_mask, stamp_max=127, miss_count=2
        )
        self.assertEqual(victims, [1, 2])
        self.assertNotIn(0, victims)
        self.assertEqual(survivors, [(71, 3), (50, 0)])


if __name__ == "__main__":
    unittest.main()
