"""German university enrollment registers, 1800-1848.

Pipeline, in build order (see the Makefile):

    tgn.load              Getty TGN relational release -> parquet
    registers             per-school register extractions -> one student table
    geocode               hometown strings -> TGN places
    polity.hometowns      hometown -> historical state/province, guild status
    linkage               records -> enrollment spells -> students
    finalize              nobility, field groups, father occupations, link ids
    polity.universities   university polity, same_state / same_polity
    gravity.build_od      origin-destination grids
    analysis.*            estimates, tables and figures
"""
