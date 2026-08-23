from kalachakra.ephemeris import se1_files as se


def test_default_span_needs_18_blocks_36_files():
    blocks = se.blocks_for_years()
    assert len(blocks) == 18
    files = se.filenames_for_years()
    assert len(files) == 36  # 18 planet + 18 moon


def test_expected_tags_in_order():
    tags = [b.tag for b in se.blocks_for_years()]
    assert tags == ["m36", "m30", "m24", "m18", "m12", "m06",
                    "_00", "_06", "_12", "_18", "_24", "_30",
                    "_36", "_42", "_48", "_54", "_60", "_66"]


def test_endpoints_are_covered():
    blocks = se.blocks_for_years()
    # 3102 BCE == astro -3101 falls in the first block (m36: 3601..3002 BCE).
    first = blocks[0]
    assert first.tag == "m36" and first.start_year <= -3101 <= first.end_year
    # 7154 CE falls in the last block (_66: 6600..7199 CE).
    last = blocks[-1]
    assert last.tag == "_66" and last.start_year <= 7154 <= last.end_year


def test_filenames_are_planets_and_moon_only():
    files = se.filenames_for_years()
    assert "seplm36.se1" in files and "sepl_66.se1" in files
    assert "semom36.se1" in files and "semo_66.se1" in files
    assert not any(f.startswith("seas") for f in files)  # no asteroids


def test_block_anchors_match_docs():
    by_tag = {b.tag: b for b in se.all_blocks()}
    assert (by_tag["_00"].start_year, by_tag["_00"].end_year) == (0, 599)
    assert (by_tag["_18"].start_year, by_tag["_18"].end_year) == (1800, 2399)
    # m36 == 3601 BCE .. 3002 BCE  (astro -3600 .. -3001)
    assert (by_tag["m36"].start_year, by_tag["m36"].end_year) == (-3600, -3001)


def test_fmt_year():
    assert se.fmt_year(2024) == "2024 CE"
    assert se.fmt_year(0) == "1 BCE"
    assert se.fmt_year(-3101) == "3102 BCE"
